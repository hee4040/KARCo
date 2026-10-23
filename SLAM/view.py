import sys, numpy as np, open3d as o3d
p = o3d.io.read_point_cloud(sys.argv[1] if len(sys.argv) > 1 else "1006test.ply")
a = np.asarray(p.points)
lo, hi = np.percentile(a, [2, 98], axis=0)  # ponytail: percentile crop, tune if the scene is clipped
pad = (hi - lo) * 0.2
p = p.crop(o3d.geometry.AxisAlignedBoundingBox(lo - pad, hi + pad))
p, _ = p.remove_statistical_outlier(20, 1.5)
o3d.visualization.draw_geometries([p])
