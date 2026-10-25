"""Standalone implementation of the validated Envision notebook Darcy model."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import asdict, dataclass
import warnings
import numpy as np
import pygimli as pg
import pyvista as pv
from scipy.ndimage import label as label_components
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

class ModelValidationError(ValueError):
    """A geometry or hydraulic validation failed; do not publish this result."""

HYDRAULIC_MATERIALS = (
    ("Bedrock matrix", "bedrock_k", "bedrock_porosity"),
    ("Aquiclude", "aquiclude_k", "aquiclude_porosity"),
    ("Deep consolidated sands", "deep_sand_k", "deep_sand_porosity"),
    ("Thin aquitard", "aquitard_k", "aquitard_porosity"),
    ("Compacted sands", "compacted_sand_k", "compacted_sand_porosity"),
    ("Loose sands / recharge strip", "loose_sand_k", "loose_sand_porosity"),
    ("Clay lens", "clay_k", "clay_porosity"),
    ("Fracture network", "fracture_k", "fracture_porosity"),
    ("Borehole conduit", "pipe_k", "pipe_porosity"),
)

@dataclass(frozen=True)
class ModelConfig:
    spring_head: float = 40.0
    head_difference: float = 10.0
    strip_width: float = 2.0
    fracture_width: float = 0.5
    bedrock_k: float = 1e-09
    fracture_k: float = 0.0002
    loose_sand_k: float = 0.0001
    pipe_k: float = 0.01
    aquiclude_k: float = 1e-9
    deep_sand_k: float = 3e-4
    aquitard_k: float = 1e-8
    compacted_sand_k: float = 2e-4
    clay_k: float = 1e-8
    bedrock_porosity: float = 0.005
    aquiclude_porosity: float = 0.10
    deep_sand_porosity: float = 0.20
    aquitard_porosity: float = 0.15
    compacted_sand_porosity: float = 0.25
    loose_sand_porosity: float = 0.30
    clay_porosity: float = 0.15
    fracture_porosity: float = 0.05
    pipe_porosity: float = 1.0
    nx: int = 50
    ny: int = 6

    def validate(self) -> None:
        values = np.array(list(asdict(self).values()), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ModelValidationError('Every parameter must be finite.')
        if self.head_difference <= 0:
            raise ModelValidationError('Recharge head must exceed the spring/river head.')
        if not 0.5 <= self.strip_width <= 4.0:
            raise ModelValidationError('Strip width must be between 0.5 and 4 m, clear of borehole 1.')
        if not 0.1 <= self.fracture_width <= 0.75:
            raise ModelValidationError('Fracture width must be between 0.1 and 0.75 m.')
        for name, k_field, porosity_field in HYDRAULIC_MATERIALS:
            if getattr(self, k_field) <= 0:
                raise ModelValidationError(f'{name} conductivity must be strictly positive.')
            if not 0 < getattr(self, porosity_field) <= 1:
                raise ModelValidationError(f'{name} effective porosity must be greater than zero and at most 1.')
        if not isinstance(self.nx, int) or not isinstance(self.ny, int):
            raise ModelValidationError('Mesh subdivisions must be integers.')
        if not 20 <= self.nx <= 80 or not 4 <= self.ny <= 10:
            raise ModelValidationError('Supported background mesh: NX=20-80 and NY=4-10.')

@dataclass
class ModelResult:
    config: ModelConfig
    grid: pv.UnstructuredGrid
    summary: dict
    active_nodes: np.ndarray
    messages: tuple[str, ...]

def horizons(x):
    """Seven original notebook interfaces in metres, bottom to terrain."""
    u = np.asarray(x) / 100.0
    top = 60.0 * np.interp(u, TOPO_X, TOPO_Z)
    rock = 12 + 12 * u
    return np.array([np.zeros_like(u), rock, rock + 1.8, 24 + 13 * u, 25.6 + 13 * u, 27 + 24 * u, top])
TOPO_X = np.array([0, 8, 10, 14, 18, 26, 29, 40, 43, 48, 53, 58, 62, 100]) / 100
TOPO_Z = np.array([30, 30, 40, 40, 44, 44, 47, 47, 59, 51, 47, 51, 60, 60]) / 60

def run_model(config: ModelConfig, progress: Callable[[str], None] | None=None) -> ModelResult:
    config.validate()
    messages = []

    def report(message: str) -> None:
        if progress is not None:
            progress(message)
    report('Preparing conforming geometry')
    L, W, H = (100.0, 20.0, 60.0)
    HEAD_SPRING = config.spring_head
    HEAD_DIFFERENCE = config.head_difference
    HEAD_RECHARGE = HEAD_SPRING + HEAD_DIFFERENCE
    SPRING_X, RECHARGE_X = ((10.0, 14.0), (90.0, 100.0))
    SPRING_Y, RECHARGE_Y = ((0.0, W), (0.0, W))
    NX, NY = (config.nx, config.ny)
    NZ_PER_LAYER = [6, 2, 5, 2, 6, 8]
    MATERIALS = {
        1: dict(name='Bedrock matrix', K=config.bedrock_k, porosity=config.bedrock_porosity, color='#789bc2'),
        2: dict(name='Aquiclude', K=config.aquiclude_k, porosity=config.aquiclude_porosity, color='#4949ca'),
        3: dict(name='Deep consolidated sands', K=config.deep_sand_k, porosity=config.deep_sand_porosity, color='#b9805c'),
        4: dict(name='Thin aquitard', K=config.aquitard_k, porosity=config.aquitard_porosity, color='#642519'),
        5: dict(name='Compacted sands', K=config.compacted_sand_k, porosity=config.compacted_sand_porosity, color='#eee2ad'),
        6: dict(name='Loose sands', K=config.loose_sand_k, porosity=config.loose_sand_porosity, color='#ffe928'),
        7: dict(name='Clay lens', K=config.clay_k, porosity=config.clay_porosity, color='#747b72'),
        8: dict(name='Fracture network', K=config.fracture_k, porosity=config.fracture_porosity, color='#1a5967'),
        9: dict(name='Right recharge strip', K=config.loose_sand_k, porosity=config.loose_sand_porosity, color='#e79d16'),
        10: dict(name='Impermeable casing', K=0.0, porosity=0.0, color='#30343b'),
        11: dict(name='Borehole water conduit', K=config.pipe_k, porosity=config.pipe_porosity, color='#00b4e8'),
    }
    LENS_X, LENS_Y, LENS_FRACTION = ((62.0, 74.0), (0.0, W), (0.5, 0.75))
    STRIP_X = (L - config.strip_width, L)
    PIPE_WIDTH, CASING_THICKNESS, WELL_Y = (1.0, 0.5, W / 2)
    FRACTURE_WIDTH, FRACTURE_SPACING = (config.fracture_width, 10.0)
    FRACTURE_REFERENCE_THICKNESS, FRACTURE_VERTICAL_SPACING = (18.0, 4.0)
    WELLS = [('9', 0.16, 0.06), ('8', 0.2, 0.32), ('7', 0.31, 0.1), ('6', 0.35, 0.37), ('5', 0.39, 0.55), ('4', 0.43, 0.76), ('3', 0.84, 0.72), ('2', 0.89, 0.45), ('1', 0.94, 0.08)]
    if not (HEAD_DIFFERENCE > 0 and L > 0 and (W > 0) and (H > 0)):
        raise ModelValidationError('Model check failed: HEAD_DIFFERENCE > 0 and L > 0 and (W > 0) and (H > 0)')
    if not (len(NZ_PER_LAYER) == 6 and min(NZ_PER_LAYER) > 0):
        raise ModelValidationError('Model check failed: len(NZ_PER_LAYER) == 6 and min(NZ_PER_LAYER) > 0')
    if not 0 < FRACTURE_WIDTH < min(FRACTURE_SPACING, FRACTURE_VERTICAL_SPACING):
        raise ModelValidationError('Model check failed: 0 < FRACTURE_WIDTH < min(FRACTURE_SPACING, FRACTURE_VERTICAL_SPACING)')
    if not (PIPE_WIDTH > 0 and CASING_THICKNESS > 0):
        raise ModelValidationError('Model check failed: PIPE_WIDTH > 0 and CASING_THICKNESS > 0')
    if not all((v['K'] > 0 and 0 < v['porosity'] <= 1 for i, v in MATERIALS.items() if i != 10)):
        raise ModelValidationError("Model check failed: all((v['K'] > 0 and 0 < v['porosity'] <= 1 for i, v in MATERIALS.items() if i != 10))")
    if not (MATERIALS[10]['K'] == 0 and MATERIALS[10]['porosity'] == 0):
        raise ModelValidationError("Model check failed: MATERIALS[10]['K'] == 0 and MATERIALS[10]['porosity'] == 0")
    half_pipe = PIPE_WIDTH / 2
    half_outer = half_pipe + CASING_THICKNESS
    bedrock_wells = [label for label, xf, zf in WELLS if zf * H < horizons(xf * L)[1]]
    fracture_x = np.unique(np.r_[np.arange(FRACTURE_SPACING, L, FRACTURE_SPACING), [xf * L for label, xf, zf in WELLS if label in bedrock_wells]])
    fracture_y = np.arange(FRACTURE_SPACING, W, FRACTURE_SPACING)
    fracture_fractions = np.arange(FRACTURE_VERTICAL_SPACING, FRACTURE_REFERENCE_THICKNESS, FRACTURE_VERTICAL_SPACING) / FRACTURE_REFERENCE_THICKNESS
    well_edges = np.array([xf * L + d for _, xf, _ in WELLS for d in [-half_outer, -half_pipe, half_pipe, half_outer]])
    xv = np.unique(np.round(np.r_[np.linspace(0, L, NX + 1), TOPO_X * L, SPRING_X, RECHARGE_X, LENS_X, STRIP_X, well_edges, fracture_x - FRACTURE_WIDTH / 2, fracture_x + FRACTURE_WIDTH / 2], 10))
    yv = np.unique(np.round(np.r_[np.linspace(0, W, NY + 1), SPRING_Y, RECHARGE_Y, LENS_Y, WELL_Y + np.array([-half_outer, -half_pipe, half_pipe, half_outer]), fracture_y - FRACTURE_WIDTH / 2, fracture_y + FRACTURE_WIDTH / 2], 10))
    if not (xv.min() == 0 and xv.max() == L and (yv.min() == 0) and (yv.max() == W)):
        raise ModelValidationError('Model check failed: xv.min() == 0 and xv.max() == L and (yv.min() == 0) and (yv.max() == W)')
    if not (np.all(np.diff(xv) > 1e-08) and np.all(np.diff(yv) > 1e-08)):
        raise ModelValidationError('Model check failed: np.all(np.diff(xv) > 1e-08) and np.all(np.diff(yv) > 1e-08)')
    if not np.all(np.diff(horizons(xv), axis=0) > 0):
        raise ModelValidationError('Layer crossing.')
    if not 0 < STRIP_X[0] < STRIP_X[1] == L:
        raise ModelValidationError('Model check failed: 0 < STRIP_X[0] < STRIP_X[1] == L')
    for label, xf, zf in WELLS:
        if not half_outer < xf * L < STRIP_X[0] - half_outer:
            raise ModelValidationError('Model check failed: half_outer < xf * L < STRIP_X[0] - half_outer')
        if not 0 < zf * H < horizons(xf * L)[-1]:
            raise ModelValidationError('Model check failed: 0 < zf * H < horizons(xf * L)[-1]')
    if not half_outer < WELL_Y < W - half_outer:
        raise ModelValidationError('Model check failed: half_outer < WELL_Y < W - half_outer')
    report('Building mesh and checking hydraulic connectivity')
    hsx = horizons(xv)
    tracks = []
    for k, n in enumerate(NZ_PER_LAYER):
        for f in np.linspace(0, 1, n + 1)[:-1]:
            tracks.append(hsx[k] + f * (hsx[k + 1] - hsx[k]))
    tracks.append(hsx[-1])
    for f in fracture_fractions:
        tracks.extend([f * hsx[1] - FRACTURE_WIDTH / 2, f * hsx[1] + FRACTURE_WIDTH / 2])
    tracks.sort(key=lambda z: float(z[len(xv) // 2]))
    ztracks = np.column_stack(tracks)
    if not np.all(np.diff(ztracks, axis=1) > 1e-08):
        raise ModelValidationError('Crossing fracture/layer tracks.')
    for label, xf, zf in WELLS:
        zline = np.array([np.interp(xf * L, xv, z) for z in ztracks.T])
        i = np.searchsorted(zline, zf * H) - 1
        alpha = (zf * H - zline[i]) / (zline[i + 1] - zline[i])
        if 1e-08 < alpha < 1 - 1e-08:
            ztracks = np.insert(ztracks, i + 1, ztracks[:, i] + alpha * (ztracks[:, i + 1] - ztracks[:, i]), axis=1)
    if not np.all(np.diff(ztracks, axis=1) > 1e-08):
        raise ModelValidationError('Model check failed: np.all(np.diff(ztracks, axis=1) > 1e-08)')
    well_bottom_index = {}
    for label, xf, zf in WELLS:
        zline = np.array([np.interp(xf * L, xv, z) for z in ztracks.T])
        i = int(np.argmin(abs(zline - zf * H)))
        if not abs(zline[i] - zf * H) < 1e-08:
            raise ModelValidationError('Model check failed: abs(zline[i] - zf * H) < 1e-08')
        well_bottom_index[label] = i
    for f in fracture_fractions:
        for offset in [-FRACTURE_WIDTH / 2, FRACTURE_WIDTH / 2]:
            expected = f * hsx[1] + offset
            if not np.any(np.all(np.isclose(ztracks, expected[:, None], atol=1e-09, rtol=0), axis=0)):
                raise ModelValidationError('Model check failed: np.any(np.all(np.isclose(ztracks, expected[:, None], atol=1e-09, rtol=0), axis=0))')
    shape = (len(xv), len(yv), ztracks.shape[1])
    ids = np.arange(np.prod(shape)).reshape(shape)
    points = np.column_stack([np.broadcast_to(xv[:, None, None], shape).ravel(), np.broadcast_to(yv[None, :, None], shape).ravel(), np.broadcast_to(ztracks[:, None, :], shape).ravel()])
    corners = np.stack([ids[:-1, :-1, :-1], ids[1:, :-1, :-1], ids[1:, 1:, :-1], ids[:-1, 1:, :-1], ids[:-1, :-1, 1:], ids[1:, :-1, 1:], ids[1:, 1:, 1:], ids[:-1, 1:, 1:]], axis=-1)
    brick_shape = corners.shape[:-1]
    centres = points[corners].mean(axis=-2)
    cx, cy, cz = np.moveaxis(centres, -1, 0)
    hc = horizons(cx)
    brick_material = 1 + np.sum(cz[None, ...] > hc[1:6], axis=0)
    fraction = (cz - hc[5]) / (hc[6] - hc[5])
    lens = (brick_material == 6) & (cx >= LENS_X[0]) & (cx <= LENS_X[1]) & (cy >= LENS_Y[0]) & (cy <= LENS_Y[1]) & (fraction >= LENS_FRACTION[0]) & (fraction <= LENS_FRACTION[1])
    brick_material[lens] = 7
    fractured = np.zeros(brick_shape, dtype=bool)
    for x in fracture_x:
        fractured |= abs(cx - x) < FRACTURE_WIDTH / 2
    for y in fracture_y:
        fractured |= abs(cy - y) < FRACTURE_WIDTH / 2
    for f in fracture_fractions:
        fractured |= abs(cz - f * hc[1]) < FRACTURE_WIDTH / 2
    brick_material[(brick_material == 1) & fractured] = 8
    brick_material[(cx >= STRIP_X[0]) & (cx <= STRIP_X[1])] = 9
    brick_well = np.zeros(brick_shape, dtype=int)
    for label, xf, zf in WELLS:
        core_xy = (abs(cx - xf * L) < half_pipe) & (abs(cy - WELL_Y) < half_pipe)
        outer_xy = (abs(cx - xf * L) < half_outer) & (abs(cy - WELL_Y) < half_outer)
        above = np.arange(brick_shape[2])[None, None, :] >= well_bottom_index[label]
        shell = outer_xy & ~core_xy & above
        core = core_xy & above
        if not not np.any((core | shell) & (brick_well != 0)):
            raise ModelValidationError('Overlapping boreholes.')
        brick_material[shell] = 10
        brick_material[core] = 11
        brick_well[core | shell] = int(label)
        if not (np.any(core) and np.any(shell)):
            raise ModelValidationError('Model check failed: np.any(core) and np.any(shell)')
        occupied = np.where(np.any(core, axis=2))
        actual_width_x = xv[occupied[0].max() + 1] - xv[occupied[0].min()]
        actual_width_y = yv[occupied[1].max() + 1] - yv[occupied[1].min()]
        if not (np.isclose(actual_width_x, PIPE_WIDTH) and np.isclose(actual_width_y, PIPE_WIDTH)):
            raise ModelValidationError('Model check failed: np.isclose(actual_width_x, PIPE_WIDTH) and np.isclose(actual_width_y, PIPE_WIDTH)')
        for axis in [0, 1]:
            for direction in [-1, 1]:
                neighbour = np.roll(brick_material, direction, axis=axis)
                shifted_core = np.roll(core, direction, axis=axis)
                if not np.all(neighbour[core & ~shifted_core] == 10):
                    raise ModelValidationError('Uncased pipe sidewall.')
    network_components, ncomponents = label_components(np.isin(brick_material, [8, 9]))
    network_id = np.unique(network_components[brick_material == 9])
    if not (len(network_id) == 1 and network_id[0] > 0):
        raise ModelValidationError('Model check failed: len(network_id) == 1 and network_id[0] > 0')
    network_id = int(network_id[0])
    if not np.all(network_components[brick_material == 8] == network_id):
        raise ModelValidationError('Disconnected fractures.')
    for label in bedrock_wells:
        inlet = (brick_well[:, :, well_bottom_index[label]] == int(label)) & (brick_material[:, :, well_bottom_index[label]] == 11)
        below = network_components[:, :, well_bottom_index[label] - 1]
        if not np.any(below[inlet] == network_id):
            raise ModelValidationError(f'Borehole {label} misses the fracture network.')
    pattern = np.array([(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)])
    tets = corners.reshape(-1, 8)[:, pattern].reshape(-1, 4)
    det = np.linalg.det(points[tets[:, 1:]] - points[tets[:, 0, None]])
    if not np.all(abs(det) > 1e-12):
        raise ModelValidationError('Degenerate tetrahedron.')
    flipped = det < 0
    tets[flipped, 1:3] = tets[flipped, 2:0:-1]
    markers = np.repeat(brick_material.ravel(), 6)
    well_ids = np.repeat(brick_well.ravel(), 6)
    if not set(markers) == set(MATERIALS):
        raise ModelValidationError('A material was not resolved.')
    mesh = pg.Mesh(3)
    for p in points:
        mesh.createNode(p)
    for t, m in zip(tets, markers):
        mesh.createTetrahedron(*[mesh.node(int(j)) for j in t], marker=int(m))
    mesh.createNeighborInfos()
    patch_heads = {101: HEAD_SPRING, 102: HEAD_RECHARGE}
    patch_names = {101: 'Spring/river surface', 102: 'Recharge surface'}
    for label, _, _ in WELLS:
        patch = 200 + int(label)
        patch_heads[patch] = HEAD_RECHARGE if int(label) <= 3 else HEAD_SPRING
        patch_names[patch] = f'Borehole {label}'
    patch_nodes = {m: set() for m in patch_heads}
    patch_area = {m: 0.0 for m in patch_heads}
    top_ids = set(ids[:, :, -1].ravel().tolist())
    for b in mesh.boundaries():
        if not b.outside():
            continue
        b.setMarker(0)
        bids = [int(n.id()) for n in b.nodes()]
        if not set(bids).issubset(top_ids):
            continue
        cell = b.leftCell()
        if cell is None:
            cell = b.rightCell()
        material = markers[cell.id()]
        if material == 10:
            continue
        x, y, _ = np.asarray(b.center())
        marker = 0
        if material == 11:
            marker = 200 + int(well_ids[cell.id()])
        elif SPRING_X[0] <= x <= SPRING_X[1] and SPRING_Y[0] <= y <= SPRING_Y[1]:
            marker = 101
        elif RECHARGE_X[0] <= x <= RECHARGE_X[1] and RECHARGE_Y[0] <= y <= RECHARGE_Y[1]:
            marker = 102
        if marker:
            b.setMarker(marker)
            patch_nodes[marker].update(bids)
            patch_area[marker] += b.size()
    if not all(patch_nodes.values()):
        raise ModelValidationError('An inlet/outlet boundary was not resolved.')
    fixed_sets = list(patch_nodes.values())
    if not sum(map(len, fixed_sets)) == len(set.union(*fixed_sets)):
        raise ModelValidationError('Boundary patches share nodes.')
    report('Solving saturated groundwater head')
    K = np.array([MATERIALS[int(m)]['K'] for m in markers])
    porosity = np.array([MATERIALS[int(m)]['porosity'] for m in markers])
    active_cells = K > 0
    active_nodes = np.unique(tets[active_cells])
    stiffness = pg.solver.createStiffnessMatrix(mesh, K)
    system_matrix = pg.utils.sparseMatrix2coo(stiffness).tocsr()
    fixed = np.array(sorted(set.union(*patch_nodes.values())), dtype=int)
    fixed_head = np.empty(len(points))
    for marker, nodes in patch_nodes.items():
        fixed_head[list(nodes)] = patch_heads[marker]
    if not np.all(np.isin(fixed, active_nodes)):
        raise ModelValidationError('Model check failed: np.all(np.isin(fixed, active_nodes))')
    free = np.setdiff1d(active_nodes, fixed)
    rhs = -system_matrix[free][:, fixed] @ fixed_head[fixed]
    h = np.full(len(points), np.nan)
    h[fixed] = fixed_head[fixed]
    with warnings.catch_warnings():
        warnings.simplefilter('error', MatrixRankWarning)
        h[free] = spsolve(system_matrix[free][:, free], rhs)
    if not np.all(np.isfinite(h[active_nodes])):
        raise ModelValidationError('Model check failed: np.all(np.isfinite(h[active_nodes]))')
    inactive_nodes = np.setdiff1d(np.arange(len(points)), active_nodes)
    if not np.all(np.isnan(h[inactive_nodes])):
        raise ModelValidationError('Model check failed: np.all(np.isnan(h[inactive_nodes]))')
    grad_h = np.zeros((len(tets), 3))
    grad_h[active_cells] = np.asarray(pg.solver.grad(mesh, np.nan_to_num(h)))[active_cells]
    q = -K[:, None] * grad_h
    v = np.zeros_like(q)
    v[active_cells] = q[active_cells] / porosity[active_cells, None]
    pressure_head = h - points[:, 2]
    if not (np.all(np.isfinite(q)) and np.all(q[~active_cells] == 0)):
        raise ModelValidationError('Model check failed: np.all(np.isfinite(q)) and np.all(q[~active_cells] == 0)')
    for marker, nodes in patch_nodes.items():
        if not np.max(abs(h[list(nodes)] - patch_heads[marker])) < 1e-07:
            raise ModelValidationError('Model check failed: np.max(abs(h[list(nodes)] - patch_heads[marker])) < 1e-07')
    report('Checking independent assembly and all borehole budgets')
    p = points[tets]
    D = p[:, 1:, :] - p[:, 0:1, :]
    volumes = abs(np.linalg.det(D)) / 6
    if not volumes.min() > 0:
        raise ModelValidationError('Model check failed: volumes.min() > 0')
    g123 = np.linalg.inv(D).transpose(0, 2, 1)
    g = np.concatenate([-g123.sum(axis=1, keepdims=True), g123], axis=1)
    local = K[:, None, None] * volumes[:, None, None] * np.einsum('nik,njk->nij', g, g)
    rows = np.repeat(tets, 4, axis=1).ravel()
    cols = np.tile(tets, (1, 4)).ravel()
    A = coo_matrix((local.ravel(), (rows, cols)), shape=(len(points), len(points))).tocsr()
    hh = np.nan_to_num(h)
    r = A @ hh
    pg_r = system_matrix @ hh
    if not np.allclose(r, pg_r, rtol=1e-06, atol=1e-11):
        raise ModelValidationError('Model check failed: np.allclose(r, pg_r, rtol=1e-06, atol=1e-11)')
    boundary_flows = {m: float(r[list(nodes)].sum()) for m, nodes in patch_nodes.items()}
    Qrecharge = sum((boundary_flows[m] for m, head in patch_heads.items() if head == HEAD_RECHARGE))
    Qspring = sum((boundary_flows[m] for m, head in patch_heads.items() if head == HEAD_SPRING))
    imbalance = abs(Qspring + Qrecharge) / max(abs(Qrecharge), 1e-30)
    free_residual = np.linalg.norm(r[free]) / max(np.linalg.norm(r[fixed]), 1e-30)
    if not (Qrecharge > 0 and Qspring < 0):
        raise ModelValidationError('Model check failed: Qrecharge > 0 and Qspring < 0')
    if not imbalance < 1e-05:
        raise ModelValidationError(f'Budget imbalance {imbalance:.3e}')
    if not free_residual < 1e-05:
        raise ModelValidationError(f'Free-node residual {free_residual:.3e}')
    grad_independent = np.einsum('ni,nij->nj', hh[tets], g)
    if not np.allclose(grad_h[active_cells], grad_independent[active_cells], rtol=1e-06, atol=1e-08):
        raise ModelValidationError('Model check failed: np.allclose(grad_h[active_cells], grad_independent[active_cells], rtol=1e-06, atol=1e-08)')
    if not (np.all(K[markers == 10] == 0) and np.all(q[markers == 10] == 0)):
        raise ModelValidationError('Model check failed: np.all(K[markers == 10] == 0) and np.all(q[markers == 10] == 0)')
    if not np.all(K[markers == 9] == MATERIALS[6]['K']):
        raise ModelValidationError("Model check failed: np.all(K[markers == 9] == MATERIALS[6]['K'])")
    local_reactions = np.einsum('nij,nj->ni', local, hh[tets])
    well_budgets = {}
    for label, xf, zf in WELLS:
        mask = (markers == 11) & (well_ids == int(label))
        ib = well_bottom_index[label]
        bottom_nodes = ids[:, :, ib].ravel()
        on_bottom = np.isin(tets[mask], bottom_nodes)
        bottom_flow = float(local_reactions[mask][on_bottom].sum())
        top_flow = boundary_flows[200 + int(label)]
        error = abs(top_flow + bottom_flow) / max(abs(top_flow), abs(bottom_flow), 1e-30)
        if not error < 1e-05:
            raise ModelValidationError(f'Borehole {label} top/bottom mismatch: {error:.3e}')
        if not (top_flow > 0 if int(label) <= 3 else top_flow < 0):
            raise ModelValidationError(f'Unexpected flow in borehole {label}')
        well_budgets[label] = dict(top_inflow_m3_day=top_flow * 86400, bottom_inflow_m3_day=bottom_flow * 86400, relative_balance_error=error, reservoir_head_m=patch_heads[200 + int(label)], bottom_centre_elevation_m=zf * H)
    if np.nanmin(h) < HEAD_SPRING - 1e-06 or np.nanmax(h) > HEAD_RECHARGE + 1e-06:
        messages.append('Head overshoot: inspect distorted elements and refine the mesh.')
    del p, D, g123, local, rows, cols, local_reactions
    report('Preparing browser-ready numerical dataset')
    vtkcells = np.column_stack([np.full(len(tets), 4), tets]).ravel()
    grid = pv.UnstructuredGrid(vtkcells, np.full(len(tets), pv.CellType.TETRA), points)
    grid.cell_data['Material'] = markers
    grid.cell_data['Borehole_ID'] = well_ids
    grid.cell_data['Hydraulically_active'] = active_cells.astype(np.uint8)
    grid.cell_data['K_m_s'] = K
    grid.cell_data['Effective_porosity'] = porosity
    grid.cell_data['Darcy_flux_m_s'] = q
    grid.cell_data['Pore_velocity_m_s'] = v
    grid.cell_data['Darcy_speed_m_day'] = np.linalg.norm(q, axis=1) * 86400
    grid.point_data['Head_m'] = h
    grid.point_data['Pressure_head_m'] = pressure_head
    summary = {'dimensions_m': [L, W, H], 'spring_head_m': HEAD_SPRING, 'recharge_head_m': HEAD_RECHARGE, 'spring_outflow_m3_day': float(-Qspring * 86400), 'recharge_inflow_m3_day': float(Qrecharge * 86400), 'relative_budget_imbalance': float(imbalance), 'free_node_relative_residual': float(free_residual), 'nodes': len(points), 'tetrahedra': len(tets), 'materials': MATERIALS, 'casing_only_nodes': len(inactive_nodes), 'negative_pressure_node_fraction': float(np.mean(pressure_head[active_nodes] < 0)), 'boundary_budgets': {str(m): dict(name=patch_names[m], head_m=patch_heads[m], area_m2=patch_area[m], inflow_m3_day=flow * 86400) for m, flow in boundary_flows.items()}, 'borehole_budgets': well_budgets, 'geometry': dict(strip_x_m=list(STRIP_X), strip_full_width=True, pipe_width_m=PIPE_WIDTH, casing_thickness_m=CASING_THICKNESS, well_y_m=WELL_Y, wells=[dict(label=label, x_m=xf * L, bottom_centre_m=zf * H) for label, xf, zf in WELLS], fracture_width_m=FRACTURE_WIDTH, fracture_x_m=fracture_x.tolist(), fracture_y_m=fracture_y.tolist(), fracture_horizontal_fractions=fracture_fractions.tolist(), fracture_reference_thickness_m=FRACTURE_REFERENCE_THICKNESS, bedrock_boreholes=bedrock_wells), 'checks': dict(fracture_strip_face_connected=True, bedrock_inlets_connected=True, pipe_sidewalls_cased=True, casing_K_and_flux_exactly_zero=True, all_pipe_top_bottom_budgets_passed=True, requested_flow_signs_passed=True), 'pygimli_version': pg.__version__, 'pyvista_version': pv.__version__, 'interpretation': 'Illustrative saturated Darcy model; ideal reservoir plumbing, finite-width fracture and pipe surrogates; not calibrated or mesh-converged.'}
    summary['config'] = asdict(config)
    summary['warnings'] = messages
    return ModelResult(config=config, grid=grid, summary=summary, active_nodes=active_nodes, messages=tuple(messages))
