import numpy as np
import cv2
import pyvista as pv
import fire

class IdGetter:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.n_surface = w * h# surface_pointsの個数

    def perim(self):
        return 2 * (self.w + self.h) - 4

    def _rev(self, i):
        i = i % self.perim()
        return self.perim() - i

    def top_edge(self, i):
        i = i % self.perim()
        if i < self.w:
            # top edge (left to right)
            return i
        elif i < self.w + self.h - 1:
            # right edge (top to bottom)
            row = i - self.w + 1
            return row * self.w + (self.w - 1)
        elif i < 2 * self.w + self.h - 2:
            # bottom edge (right to left)
            col = self.w - 1 - (i - self.w - self.h + 2)
            return (self.h - 1) * self.w + col
        else:
            # left edge (bottom to top)
            row = self.perim() - i
            return row * self.w

    def bottom_edge(self, i):
        return i + self.n_surface

    def top_edge_rev(self, i):
        return self.top_edge(self._rev(i))

    def bottom_edge_rev(self, i):
        return self.bottom_edge(self._rev(i))

def load_depth_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    if len(img.shape) == 2:
        depth = img.astype(np.float32)
    elif len(img.shape) == 3:
        depth = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        raise ValueError(f"Unsupported image format.{img.shape}")
    depth -= depth.min()
    if depth.max() > 0:
        depth /= depth.max()
    return depth

def create_faces(width, height):
    faces = []
    for i in range(height - 1):
        for j in range(width - 1):
            idx = lambda x, y: y * width + x
            a = idx(j, i)
            b = idx(j + 1, i)
            c = idx(j + 1, i + 1)
            d = idx(j, i + 1)
            faces.append([3, a, b, d])  # triangle 1
            faces.append([3, b, c, d])  # triangle 2
    return np.array(faces)

def create_side_walls(idgetter:IdGetter):
    faces = []
    for i in range(idgetter.perim()):
        a = idgetter.top_edge(i)
        b = idgetter.top_edge(i+1)
        c = idgetter.bottom_edge(i+1)
        d = idgetter.bottom_edge(i)
        faces.append([3, a, b, d])  # triangle 1
        faces.append([3, b, c, d])  # triangle 2
    return np.array(faces)
        
def create_back_faces(idgetter:IdGetter):
    faces = []
    faces.append([3, idgetter.bottom_edge(0), idgetter.bottom_edge(1), idgetter.bottom_edge_rev(1)])  # triangle 1
    for i in range(1, idgetter.w + idgetter.h-1):
        a = idgetter.bottom_edge(i)
        b = idgetter.bottom_edge(i + 1)
        c = idgetter.bottom_edge_rev(i + 1)
        d = idgetter.bottom_edge_rev(i)
        faces.append([3, a, b, d])  # triangle 1
        faces.append([3, b, c, d])  # triangle 2
    faces.append([3, idgetter.bottom_edge(i), idgetter.bottom_edge(i + 1), idgetter.bottom_edge_rev(i)])  # triangle 1
    return faces
        
def create_stl_from_depth(image_path, stl_path="output.stl", height=20.0, base_thickness=20.0):
    depth = load_depth_image(image_path)
    h, w = depth.shape
    z_surface = depth * height + base_thickness
    z_base = np.zeros_like(z_surface)

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    y = h - 1 - y  # flip y-axis for proper orientation

    surface_points = np.c_[x.ravel(), y.ravel(), z_surface.ravel()]
    idgetter = IdGetter(w, h)
    edge_points = (
        [[i, h-1, 0] for i in range(w)] +
        [[w-1, h-1-i, 0] for i in range(1, h)] +
        [[w-1-i, 0, 0] for i in range(1, w)] +
        [[0, i, 0] for i in range(1, h)]
    )

    all_points = np.vstack([surface_points, edge_points])

    surface_faces = create_faces(w, h)

    base_faces = create_back_faces(idgetter)

    side_faces = create_side_walls(idgetter)

    all_faces = np.vstack([surface_faces, base_faces, side_faces])

    mesh = pv.PolyData()
    mesh.points = all_points
    mesh.faces = all_faces
    mesh.clean(tolerance=1e-3)
    mesh.scale(0.1, 0.1, 0.1)
    # Compute normals (if not already present)
    mesh.flip_faces(inplace=True)
    mesh.compute_normals(inplace=True)

    # Create glyphs for the normals
    normals = mesh.glyph(orient='Normals', scale=False, factor=0.05)
    
    pl = pv.Plotter()
    pl.add_mesh(mesh, show_edges=True, cmap="coolwarm")
    pl.add_mesh(normals, color='red')
    pl.show()

    mesh.save(stl_path)
    print(f"✅ Watertight STL saved to: {stl_path}")

if __name__ == "__main__":
    fire.Fire(create_stl_from_depth)
