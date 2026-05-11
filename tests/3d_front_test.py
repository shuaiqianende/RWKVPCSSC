import open3d as o3d
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import numpy as np
import pandas as pd
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
import torch
from scipy import stats
from joblib import Parallel, delayed
from collections.abc import Iterable
from collections import OrderedDict
import sys



color_dict = {
    1: {'rgb': [1.0, 0.0, 0.0], 'name': 'Red'},
    2: {'rgb': [0.0, 1.0, 0.0], 'name': 'Green'},
    3: {'rgb': [0.0, 0.0, 1.0], 'name': 'Blue'},
    4: {'rgb': [1.0, 1.0, 0.0], 'name': 'Yellow'},
    5: {'rgb': [1.0, 0.5, 0.0], 'name': 'Orange'},
    6: {'rgb': [0.5, 0.0, 0.5], 'name': 'Purple'},
    7: {'rgb': [0.0, 1.0, 1.0], 'name': 'Cyan'},
    8: {'rgb': [1.0, 0.0, 1.0], 'name': 'Magenta'},
    9: {'rgb': [0.5, 0.5, 0.5], 'name': 'Gray'},
    10: {'rgb': [1.0, 1.0, 1.0], 'name': 'White'},
    11: {'rgb': [0.0, 0.0, 0.0], 'name': 'Black'},
    12: {'rgb': [0.8, 0.8, 0.0], 'name': 'Olive'},
    13: {'rgb': [0.8, 0.0, 0.8], 'name': 'Violet'},
    14: {'rgb': [0.0, 0.8, 0.8], 'name': 'Turquoise'},
    15: {'rgb': [0.5, 0.0, 0.0], 'name': 'Brown'},
    16: {'rgb': [0.0, 0.5, 0.0], 'name': 'Dark Green'}
}


_SUPER_CATEGORIES_3D = [
    {'id': 1 , 'category': 'Cabinet'},  # 包括 'cabinet', 'kitchen cabinet'
    {'id': 2 , 'category': 'Pier/Stool'},  # 包括 'Pier/Stool'
    {'id': 3 , 'category': 'Bed'},  # 包括 'bed'
    {'id': 4 , 'category': 'Chair'},  # 包括 'chair'
    {'id': 5 , 'category': 'Table'},  # 包括 'table', 'vanity'
    {'id': 6 , 'category': 'Sofa'},  # 包括 'sofa'
    {'id': 7 , 'category': 'Lighting'},  # 包括 'lighting'
    {'id': 8 , 'category': 'Storage'},  # 包括 'wardrobe', 'storage unit'
    {'id': 9 , 'category': 'Electric'},  # 包括 'media unit', 'electric'
    {'id': 10, 'category': 'Decor'},  # 包括 'art', 'accessory', 'mirror', 'rug'
    {'id': 11, 'category': 'Windows'},  # 包括 'window', 'curtain'
    {'id': 12, 'category': 'Door'},  # 包括 'door', 'barn door'
    {'id': 13, 'category': 'Water related'},  # 包括 'basin', 'sink', 'water'
    {'id': 14, 'category': 'Attach obj'},  # 包括 '200 - on the floor', '300 - on top of others', '400 - attach to wall', '500 - attach to ceiling', 'attachment'
    {'id': 15, 'category': 'Plants'},  # 包括 'plants'
    {'id': 16, 'category': 'Wall-related'},  # 包括 'wall-related'
    {'id': 17, 'category': 'Ceiling-related'},  # 包括 'ceiling-related'
    {'id': 18, 'category': 'Floor-related'},  # 包括 'floor-related'
    {'id': 19, 'category': 'Structure and Openings'},  # 包括 'structure and openings'
    {'id': 20, 'category': 'Pipe'},
    {'id': 21, 'category': 'stair'},
    {'id': 22, 'category': 'Others'},
    # {'id': 20, 'category': 'Edges and Trim'},
]
_MESH_CATEGORIES_3D = [
    {'id': 0, 'super-category': 'Wall-related', 'category': 'WallBottom'},
    {'id': 1, 'super-category': 'Wall-related', 'category': 'WallInner'},
    {'id': 2, 'super-category': 'Wall-related', 'category': 'WallOuter'},
    {'id': 3, 'super-category': 'Wall-related', 'category': 'WallTop'},
    {'id': 4, 'super-category': 'Wall-related', 'category': 'SlabSide'},
    {'id': 5, 'super-category': 'Wall-related', 'category': 'SlabBottom'},
    {'id': 6, 'super-category': 'Wall-related', 'category': 'SlabTop'},
    {'id': 7, 'super-category': 'Wall-related', 'category': 'Column'},
    {'id': 8, 'super-category': 'Ceiling-related', 'category': 'Ceiling'},
    {'id': 9, 'super-category': 'Ceiling-related', 'category': 'Beam'},
    {'id': 10, 'super-category': 'Floor-related', 'category': 'Floor'},
    {'id': 11, 'super-category': 'Wall-related', 'category': 'CustomizedBackgroundModel'},
    {'id': 12, 'super-category': 'Wall-related', 'category': 'Customized_wainscot'},
    {'id': 13, 'super-category': 'Wall-related', 'category': 'CustomizedFeatureWall'},
    {'id': 14, 'super-category': 'Wall-related', 'category': 'ExtrusionCustomizedBackgroundWall'},
    {'id': 15, 'super-category': 'Ceiling-related', 'category': 'SmartCustomizedCeiling'},
    {'id': 16, 'super-category': 'Ceiling-related', 'category': 'CustomizedCeiling'},
    {'id': 17, 'super-category': 'Ceiling-related', 'category': 'ExtrusionCustomizedCeilingModel'},
    {'id': 18, 'super-category': 'Structure and Openings', 'category': 'Door'},
    {'id': 19, 'super-category': 'Structure and Openings', 'category': 'Window'},
    {'id': 20, 'super-category': 'Structure and Openings', 'category': 'Hole'},
    {'id': 21, 'super-category': 'Structure and Openings', 'category': 'Pocket'},
    {'id': 22, 'super-category': 'Structure and Openings', 'category': 'BayWindow'},
    {'id': 23, 'super-category': 'Cabinet', 'category': 'Cabinet/LightBand'},
    {'id': 24, 'super-category': 'Cabinet', 'category': 'Cabinet'},
    {'id': 25, 'super-category': 'Lighting', 'category': 'LightBand'},
    {'id': 26, 'super-category': 'Pipe', 'category': 'Flue'},
    {'id': 27, 'super-category': 'Pipe', 'category': 'SewerPipe'},
]

_FURNITURE_CATEGORIES_3D = [
    {'id': 0, 'super-category': 'Sofa', 'category': 'sofa'},
    {'id': 1, 'super-category': 'Bed', 'category': 'bed'},
    {'id': 2, 'super-category': 'Chair', 'category': 'chair'},
    {'id': 3, 'super-category': 'Lighting', 'category': 'lighting'},
    {'id': 4, 'super-category': 'Table', 'category': 'table'},
    {'id': 5, 'super-category': 'Table', 'category': 'vanity'},
    {'id': 6, 'super-category': 'Cabinet', 'category': 'cabinet'},
    {'id': 7, 'super-category': 'Cabinet', 'category': 'kitchen cabinet'},
    {'id': 8, 'super-category': 'Storage', 'category': 'wardrobe'},
    {'id': 9, 'super-category': 'Storage', 'category': 'storage unit'},
    {'id': 10, 'super-category': 'Electric', 'category': 'media unit'},
    {'id': 11, 'super-category': 'Storage', 'category': 'shelf'},
    {'id': 12, 'super-category': 'Decor', 'category': 'art'},
    {'id': 13, 'super-category': 'Decor', 'category': 'accessory'},
    {'id': 14, 'super-category': 'Decor', 'category': 'mirror'},
    {'id': 15, 'super-category': 'Decor', 'category': 'rug'},
    {'id': 16, 'super-category': 'Windows', 'category': 'window'},
    {'id': 17, 'super-category': 'Windows', 'category': 'curtain'},
    {'id': 18, 'super-category': 'Door', 'category': 'door'},
    {'id': 19, 'super-category': 'Door', 'category': 'barn door'},
    {'id': 20, 'super-category': 'Water related', 'category': 'basin'},
    {'id': 21, 'super-category': 'Water related', 'category': 'sink'},
    {'id': 22, 'super-category': 'Water related', 'category': 'water'},
    {'id': 23, 'super-category': 'Water related', 'category': 'shower'},
    {'id': 24, 'super-category': 'Water related', 'category': 'bath'},
    {'id': 25, 'super-category': 'Attach obj', 'category': '200 - on the floor'},
    {'id': 26, 'super-category': 'Attach obj', 'category': '300 - on top of others'},
    {'id': 27, 'super-category': 'Attach obj', 'category': '400 - attach to wall'},
    {'id': 28, 'super-category': 'Attach obj', 'category': '500 - attach to ceiling'},
    {'id': 29, 'super-category': 'Attach obj', 'category': 'attachment'},
    {'id': 30, 'super-category': 'Electric', 'category': 'electric'},
    {'id': 31, 'super-category': 'Electric', 'category': 'electronics'},
    {'id': 32, 'super-category': 'Plants', 'category': 'plants'},
    {'id': 33, 'super-category': 'Others', 'category': 'Other'},

    {'id': 34, 'super-category': 'Cabinet', 'category': 'Children Cabinet'},
    {'id': 35, 'super-category': 'Cabinet', 'category': 'Nightstand'},
    {'id': 36, 'super-category': 'Cabinet', 'category': 'Bookcase / jewelry Armoire'},
    {'id': 38, 'super-category': 'Table', 'category': 'Coffee Table'},
    {'id': 39, 'super-category': 'Table', 'category': 'Corner/Side Table'},
    {'id': 40, 'super-category': 'Table', 'category': 'Sideboard / Side Cabinet / Console Table'},
    {'id': 41, 'super-category': 'Cabinet', 'category': 'Wine Cabinet'},
    {'id': 42, 'super-category': 'Cabinet', 'category': 'TV Stand'},
    {'id': 43, 'super-category': 'Cabinet', 'category': 'Drawer Chest / Corner cabinet'},
    {'id': 44, 'super-category': 'Cabinet', 'category': 'Shelf'},
    {'id': 45, 'super-category': 'Table', 'category': 'Round End Table'},
    {'id': 46, 'super-category': 'Bed', 'category': 'King-size Bed'},
    {'id': 47, 'super-category': 'Bed', 'category': 'Bunk Bed'},
    {'id': 48, 'super-category': 'Bed', 'category': 'Bed Frame'},
    {'id': 49, 'super-category': 'Bed', 'category': 'Single bed'},
    {'id': 50, 'super-category': 'Bed', 'category': 'Kids Bed'},
    {'id': 51, 'super-category': 'Chair', 'category': 'Dining Chair'},
    {'id': 52, 'super-category': 'Chair', 'category': 'Lounge Chair / Cafe Chair / Office Chair'},
    {'id': 53, 'super-category': 'Chair', 'category': 'Dressing Chair'},
    {'id': 54, 'super-category': 'Chair', 'category': 'Classic Chinese Chair'},
    {'id': 55, 'super-category': 'Chair', 'category': 'Barstool'},
    {'id': 56, 'super-category': 'Table', 'category': 'Dressing Table'},
    {'id': 57, 'super-category': 'Table', 'category': 'Dining Table'},
    {'id': 58, 'super-category': 'Table', 'category': 'Desk'},
    {'id': 59, 'super-category': 'Sofa', 'category': 'Three-Seat / Multi-seat Sofa'},
    {'id': 60, 'super-category': 'Sofa', 'category': 'armchair'},
    {'id': 61, 'super-category': 'Sofa', 'category': 'Loveseat Sofa'},
    {'id': 62, 'super-category': 'Sofa', 'category': 'L-shaped Sofa'},
    {'id': 63, 'super-category': 'Sofa', 'category': 'Lazy Sofa'},
    {'id': 64, 'super-category': 'Sofa', 'category': 'Chaise Longue Sofa'},
    {'id': 65, 'super-category': 'Pier/Stool', 'category': 'Footstool / Sofastool / Bed End Stool / Stool'},
    {'id': 66, 'super-category': 'Lighting', 'category': 'Pendant Lamp'},
    {'id': 67, 'super-category': 'Lighting', 'category': 'Ceiling Lamp'},
    {'id': 68, 'super-category': 'Cabinet', 'category': 'Shoe Cabinet'},
    {'id': 69, 'super-category': 'Bed', 'category': 'Couch Bed'},
    {'id': 70, 'super-category': 'Chair', 'category': 'Hanging Chair'},
    {'id': 71, 'super-category': 'Chair', 'category': 'Folding chair'},
    {'id': 72, 'super-category': 'Table', 'category': 'Bar'},
    {'id': 73, 'super-category': 'Sofa', 'category': 'U-shaped Sofa'},
    {'id': 74, 'super-category': 'Lighting', 'category': 'Floor Lamp'},
    {'id': 75, 'super-category': 'Lighting', 'category': 'Wall Lamp'},
    {'id': 76, 'super-category': 'Sofa', 'category': 'Two-seat Sofa'},
    {'id': 77, 'super-category': 'stair', 'category': 'stair'},
    {'id': 78, 'super-category': 'Cabinet', 'category': 'Sideboard / Side Cabinet / Console'},
    {'id': 79, 'super-category': 'Others', 'category': 'unknown'},
    {'id': 80, 'super-category': 'Others', 'category': 'Unknow'},
    {'id': 81, 'super-category': 'Bed', 'category': 'Double Bed'},
    {'id': 82, 'super-category': 'Chair', 'category': 'Lounge Chair / Book-chair / Computer Chair'},
    {'id': 83, 'super-category': 'Cabinet', 'category': 'Cabinet/Shelf/Desk'},
    {'id': 84, 'super-category': 'Table', 'category': 'Tea Table'},
    {'id': 85, 'super-category': 'Pier/Stool', 'category': 'Pier/Stool'},
    {'id': 86, 'super-category': 'Electric', 'category': 'appliance'},
    {'id': 87, 'super-category': 'Sofa', 'category': 'Three-Seat / Multi-person sofa'},
    {'id': 88, 'super-category': 'Others', 'category': 'Others'},
    {'id': 89, 'super-category': 'Electric', 'category': 'meter'},
    {'id': 90, 'super-category': 'Others', 'category': 'recreation'},
    {'id': 91, 'super-category': 'Water related', 'category': 'Wine Cooler'},
    {'id': 92, 'super-category': 'Water related', 'category': 'toilet'},
]

def get_category_id(input_str):
    for item in _SUPER_CATEGORIES_3D:
        if item['category'].lower() == input_str.lower():
            return item['id']
    return 22
def find_mesh_super_category(category_str):
    for item in _MESH_CATEGORIES_3D:
        if item['category'].lower() == category_str.lower():  # 忽略大小写比较
            return item['super-category'], item['id']
    return 'None', -1

def find_furniture_super_category(category_str):
    for item in _FURNITURE_CATEGORIES_3D:
        if item['category'].lower() == category_str.lower():  # 忽略大小写比较
            return item['super-category'], item['id']
    return 'None', -1

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

base_dir = '/Path/3D-FRONT-PC/'
shapeLocalSource = '/Path/3D-FRONT-PC/3D-FUTURE-model/'

def __create_mesh_table(sceneDict):
    _ = {'id': [], 'material_id': [], 'type': [], 'xyz': [], 'normal': [], 'uv': [], 'face': []}
    for index, mesh in enumerate(sceneDict['mesh']):
        _['id'].append(mesh['uid'])
        _['material_id'].append(mesh['material'])
        _['type'].append(mesh['type'])
        _['xyz'].append(np.array(mesh['xyz']).reshape(-1, 3).tolist())
        _['normal'].append(np.array(mesh['normal']).reshape(-1, 3).tolist())
        _['uv'].append(np.array(mesh['uv']).reshape(-1, 2).tolist())
        _['face'].append(np.array(mesh['faces']).reshape(-1, 3).tolist())
    return {key: value for (key, value) in _.items()}
    # return {key: np.array(value) for (key, value) in _.items()}
def __create_furniture_table(sceneDict):
    _ = {'id': [], 'jid': [], 'title': []}
    for furniture in sceneDict['furniture']:
        title = furniture.get('title')
        if title != None and title != '':
            title = title.split('/')[0]
        elif title == None or title == '':
            title = furniture.get('category')
        if title == None:
            continue
        _['title'].append(title)
        _['id'].append(furniture['uid'])
        _['jid'].append(furniture['jid'])
    return {key: np.array(value) for key, value in _.items()}

def __create_instance_table(sceneDict):
    _ = {'id': [], 'position': [], 'rotation': [], 'scale': [], 'ref': [], 'room_Id': []}
    room_Id = []
    for room in sceneDict['scene']['room']:
        room_Id.append(room['instanceid'])  # 在循环内添加 instanceid
        for child in room['children']:
            _['id'].append(child['instanceid'])
            _['position'].append(child['pos'])
            _['rotation'].append(child['rot'])
            _['scale'].append(child['scale'])
            _['ref'].append(child['ref'])
            _['room_Id'].append(room['instanceid'])
            # types.add(child['type'])
    return {key: np.array(value) for key, value in _.items()}, room_Id

def save_mesh_to_obj(filename, xyz, normals, faces):
    with open(filename, 'w') as f:
        for i in range(xyz.shape[1]):
            f.write(f'v {xyz[0, i]} {xyz[1, i]} {xyz[2, i]}\n')
        for normal in normals.T:
            f.write(f'vn {normal[0]} {normal[1]} {normal[2]}\n')
        for face in faces.T:
            f.write(f'f {face[0] + 1}//{face[0] + 1} {face[1] + 1}//{face[1] + 1} {face[2] + 1}//{face[2] + 1}\n')
    print(f"Mesh saved to {filename}")


def create_triangle_meshes(mesh_data: dict, room_Id) -> list:
    meshes = []
    for i in range(len(mesh_data['xyz'])):
    # for key, row in mesh_data.items():  # 遍历字典中的每个键值对
        mesh = o3d.geometry.TriangleMesh()
        xyz = np.array(mesh_data['xyz'][i])  # 顶点坐标
        faces = np.array(mesh_data['face'][i])  # 面索引，转置
        normals = np.array(mesh_data['normal'][i])  # 顶点法线
        mesh.vertices = o3d.utility.Vector3dVector(xyz)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        num_vertices = len(mesh.vertices)
        super_category, son_id = find_mesh_super_category(mesh_data['type'][i])
        if super_category == 'None':
            continue
        if son_id in [0, 2, 3]:
            continue
        category_id = get_category_id(super_category)
        # if category_id == 10:
        #     continue
        idx = room_Id.index(mesh_data['room_Id'][i])
        color_value = np.array([[category_id / 255, idx / 255, 0]] * num_vertices)
        mesh.vertex_colors = o3d.utility.Vector3dVector(color_value.astype(np.float64))
        meshes.append(mesh)
    return meshes
def merge_triangle_meshes(meshes: list) -> o3d.geometry.TriangleMesh:
    merged_mesh = o3d.geometry.TriangleMesh()
    all_vertices = []
    all_faces = []
    all_normals = []
    all_colors = []
    vertex_offset = 0  # 记录顶点偏移量，用于更新面片的索引
    for mesh in meshes:
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.vertex_normals)
        colors = np.asarray(mesh.vertex_colors)
        all_vertices.append(vertices)
        all_normals.append(normals)
        all_faces.append(faces + vertex_offset)
        all_colors.append(colors)
        vertex_offset += vertices.shape[0]
    merged_mesh.vertices = o3d.utility.Vector3dVector(np.vstack(all_vertices))
    merged_mesh.triangles = o3d.utility.Vector3iVector(np.vstack(all_faces))
    merged_mesh.vertex_normals = o3d.utility.Vector3dVector(np.vstack(all_normals))
    merged_mesh.vertex_colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    return merged_mesh

def _get_pcd_from_mesh_o3d(mesh_o3d):
    # pts_number = 4096 * 32
    # factor = 3
    points_per_triangle = 2000
    triangles_data = extract_vertices_and_triangles(mesh_o3d)
    _xyz, _labels = generate_point_cloud(triangles_data, points_per_triangle)
    # if not mesh_o3d.has_vertices() or not mesh_o3d.has_triangles():
    #     return None, None, None
    # pcd_o3d = mesh_o3d.sample_points_poisson_disk(pts_number, factor)
    xyz = np.asarray(_xyz.cpu())
    labels = np.around(np.asarray(_labels.cpu())[:, 0:2] * 255)
    return xyz, labels, 1

def join(ndarray1Dict, ndarray2Dict, c1, c2, rsuffix):
    leftColumnNames = list(ndarray1Dict.keys())
    rightColumnNames = list(ndarray2Dict.keys())
    def rightName(_name):
        return _name if _name not in leftColumnNames else f'{rsuffix}{_name}'
    columnNames = leftColumnNames + [rightName(name) for name in rightColumnNames]
    dict1KeyIndex = list(ndarray1Dict.keys()).index(c1)
    result = []
    for row1 in zip(*ndarray1Dict.values()):
        indices = (ndarray2Dict[c2] == row1[dict1KeyIndex])
        row_join = [v[indices] for v in ndarray2Dict.values()]
        for row2 in zip(*row_join):
            result.append([*row1, *row2])
    columns = np.array(result, dtype=object).T.tolist()
    return {name: column for name, column in zip(columnNames, columns)}

def get_furniture_position(furniture, obj_fetcher):
    jid, position, rotation, scale = furniture
    # x, y, z, w = rotation
    # rotation = (w, x, y, z)
    obj_path = obj_fetcher(jid)
    if obj_path is None:
        return None  # 如果文件路径无效，返回 None
    # 创建单位矩阵
    scaleMatrix = np.eye(4)
    scaleMatrix[0][0] = scale[0]
    scaleMatrix[1][1] = scale[1]
    scaleMatrix[2][2] = scale[2]
    translationMatrix = np.eye(4)
    translationMatrix[0][3] = position[0]
    translationMatrix[1][3] = position[1]
    translationMatrix[2][3] = position[2]
    x, y, z, w = rotation
    rotationMatrix = np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w, 0],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w, 0],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y, 0],
        [0, 0, 0, 1]
    ])
    transformationMatrix = translationMatrix @ rotationMatrix @ scaleMatrix
    return transformationMatrix

def trans_obj(title, jids, room_Ids, room_Id, matrixs):
    obj_meshes = []
    for i, Trans_M in enumerate(matrixs):
        type = title[i] #.split('/')[0]
        # if type == '' or type == 'Other':
        #     continue
        super_category, son_id = find_furniture_super_category(type)
        if super_category == 'None':
            return
        category_id = get_category_id(super_category)
        obj_path = os.path.join(shapeLocalSource, jids[i], 'raw_model.obj')
        # error
        obj_mesh = o3d.io.read_triangle_mesh(obj_path)
        obj_mesh.transform(Trans_M)
        num_vertices = len(obj_mesh.vertices)
        if num_vertices == 0:
            continue
        idx = room_Id.index(room_Ids[i])
        color_value = np.array([[category_id / 255, idx / 255, 0]] * num_vertices)
        obj_mesh.vertex_colors = o3d.utility.Vector3dVector(color_value.astype(np.float64))
        obj_meshes.append(obj_mesh)
    return obj_meshes

def world_to_camera_coordinates(points_world, camera, camera_up=np.array([0, 1, 0])):
    camera_pos = np.array(camera['pos'])
    camera_target = np.array(camera['target'])
    camera_direction = camera_target - camera_pos
    camera_direction /= np.linalg.norm(camera_direction)
    camera_right = np.cross(camera_up, camera_direction)
    camera_right /= np.linalg.norm(camera_right)
    camera_up_corrected = np.cross(camera_direction, camera_right)
    rotation_matrix = np.vstack([camera_right, camera_up_corrected, -camera_direction]).T
    translation_vector = -rotation_matrix @ camera_pos
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    points_world_homogeneous = np.hstack([points_world, np.ones((points_world.shape[0], 1))])
    points_camera_homogeneous = points_world_homogeneous @ transformation_matrix.T
    points_camera = points_camera_homogeneous[:, :3]
    points_camera[:, 2] *= -1
    points_camera[:, 1] *= 1.333333
    return points_camera


def camera_to_world_coordinates(points_camera, camera, camera_up=np.array([0, 1, 0])):
    camera_pos = np.array(camera['pos'])
    camera_target = np.array(camera['target'])
    camera_direction = camera_target - camera_pos
    camera_direction /= np.linalg.norm(camera_direction)
    camera_right = np.cross(camera_up, camera_direction)
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(camera_direction, camera_right)
    rotation_matrix = np.vstack([camera_right, camera_up, camera_direction]).T
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix.T
    transformation_matrix[:3, 3] = camera_pos
    points_camera_homogeneous = np.hstack([points_camera, np.ones((points_camera.shape[0], 1))])
    points_world_homogeneous = points_camera_homogeneous @ transformation_matrix.T
    points_world = points_world_homogeneous[:, :3]
    return points_world

def pcd_downsample(pcd, number):
    pcd_ts = torch.tensor(
        pcd, dtype=torch.float, device=torch.device("cuda"), requires_grad=False
    )
    pcd_ts = pcd_ts.unsqueeze(0)
    ds_idx = furthest_point_sample(pcd_ts[:, :, :3].contiguous(), number)
    pcd_ds = gather_operation(pcd_ts.permute(0, 2, 1).contiguous(), ds_idx)
    pcd_ds = pcd_ds.permute(0, 2, 1).squeeze(0).cpu().numpy()
    return pcd_ds

def get_rays(camera, width_px = 640, height_px = 480):
    eye = np.array(camera["pos"])
    center = np.array(camera["target"])
    up = np.array([0, 1, 0])  # 假设上方向是Y轴
    fov_deg = camera["fov"]
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=fov_deg,
        center=center.tolist(),  # 转换为列表
        eye=eye.tolist(),        # 转换为列表
        up=up.tolist(),          # 转换为列表
        width_px=width_px,
        height_px=height_px
    )
    return rays

def is_point_in_frustum(point, camera_pos, camera_dir, fov, near=0.1, far=100):
    to_point = point - camera_pos
    distance = np.linalg.norm(to_point)
    if distance < near or distance > far:
        return False
    to_point_normalized = to_point / distance
    camera_dir_normalized = camera_dir / np.linalg.norm(camera_dir)
    half_fov = fov / 2.0
    angle = np.arccos(np.clip(np.dot(camera_dir_normalized, to_point_normalized), -1.0, 1.0))
    return angle <= half_fov

def get_input_crop(pcd):
    Z = pcd[:, 2]
    near = max(0.2, Z.min())  # 取场景中最近的物体距离
    far = min(6, Z.max()+0.02)  # 取场景中最远的物体距离
    valid_depth = (Z > near) & (Z < far)
    pcd_crop = pcd[valid_depth]
    return pcd_crop, (near, far)

def get_gt_in_camera_view(gt_points_camera, n_f, fov_deg):
    (near, far) = n_f
    fov_rad = np.deg2rad(fov_deg)
    half_fov_tan = np.tan(fov_rad / 2)
    Z = gt_points_camera[:, 2]
    valid_depth = (Z > near) & (Z < far)
    visible_gt_points = gt_points_camera[valid_depth]
    X, Y, Z = visible_gt_points[:, 0], visible_gt_points[:, 1], visible_gt_points[:, 2]
    frustum_x_bound = half_fov_tan * Z
    frustum_y_bound = half_fov_tan * Z
    in_frustum_x = np.abs(X) <= frustum_x_bound
    in_frustum_y = np.abs(Y) <= frustum_y_bound
    visible_mask = in_frustum_x & in_frustum_y
    visible_gt_points = visible_gt_points[visible_mask]
    return visible_gt_points

def setup_camera_params(camera, width=640, height=480):
    pos = np.array(camera['pos'])
    target = np.array(camera['target'])
    fov = camera['fov']
    direction = target - pos
    direction /= np.linalg.norm(direction)  # 单位化
    up = np.array([0, 1, 0])  # 假设上方向是Y轴
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = np.vstack((right, up, -direction)).T  # Z轴方向取反
    extrinsics[:3, 3] = pos
    fx = (width / 2) / np.tan(np.radians(fov) / 2)
    fy = (height / 2) / np.tan(np.radians(fov) / 2)
    cx, cy = width / 2, height / 2
    intrinsics = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]])
    return intrinsics, extrinsics


def depth_to_point_cloud(depth_image, label_map, intrinsics):
    _label = label_map[:, :, 0]
    _room_Id = label_map[:, :, 1]
    height, width = depth_image.shape
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    Z = depth_image.flatten()
    L = _label.flatten()
    R = _room_Id.flatten()
    valid = (Z > 0) & np.isfinite(Z)
    Z, L, R = Z[valid], L[valid], R[valid]
    X = (X.flatten()[valid] - intrinsics[0, 2]) * Z / intrinsics[0, 0]
    Y = (Y.flatten()[valid] - intrinsics[1, 2]) * Z / intrinsics[1, 1]
    most_common_value, count = stats.mode(R, keepdims=False)
    mask = (R == most_common_value)
    R, L, X, Y, Z = R[mask], L[mask], X[mask], Y[mask], Z[mask]
    # points = np.vstack((X, Y, Z, L)).T  # 组合成 N x 3 的点云
    points = np.vstack((X, Y, Z, L, R)).T  # 组合成 N x 3 的点云
    return points


def transform_to_world(points_camera, extrinsics):
    pcd = o3d.geometry.PointCloud()
    points = o3d.utility.Vector3dVector(points_camera)
    pcd.points = points
    pcd.transform(extrinsics)
    points_world = np.asarray(pcd.points)
    return points_world
def get_input(scene, camera, merge_mesh):
    intrinsics, extrinsics = setup_camera_params(camera)
    rays = get_rays(camera)
    raycasting_result = scene.cast_rays(rays)
    depth_image = raycasting_result['t_hit'].numpy()
    primitive_ids_np = raycasting_result['primitive_ids'].numpy()  # 将 Tensor 转换为 NumPy 数组
    vertex_colors_np = np.asarray(merge_mesh.vertex_colors)
    triangles_idx = np.asarray(merge_mesh.triangles)
    triangles_idx = triangles_idx[:, 0]
    triangles_color = vertex_colors_np[triangles_idx]
    color_image = np.full((480, 640, 3), -1, dtype=np.float32)  # 默认背景色为 -1
    valid_mask = primitive_ids_np != 4294967295
    valid_primitive_ids = primitive_ids_np[valid_mask]
    if valid_primitive_ids.size > 0:
        color_image[valid_mask] = triangles_color[valid_primitive_ids] * 255
    pcd = depth_to_point_cloud(depth_image, color_image, intrinsics)
    pcd = np.asfortranarray(pcd, dtype=np.float32)
    if pcd.shape[0] < 8192*1.5:
        return None, None, None, None
    pcd, (near, far) = get_input_crop(pcd)
    if pcd.shape[0] < 8192:
        return None, None, None, None
    pcd = pcd_downsample(pcd, 4096)
    # input_pts = transform_to_world(input_pts, extrinsics)
    return pcd, pcd[0, 4], near, far

def group_point_clouds_by_roomid(points):
    unique_labels = np.unique(points[:, 4])
    point_clouds_dict = {}
    for label in unique_labels:
        group = points[points[:, 4] == label]
        point_clouds_dict[label] = group  # 将标签作为键，点云作为值
    return point_clouds_dict


def extract_vertices_and_triangles(mesh):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    vertices_gpu = torch.tensor(vertices, device='cuda')
    colors_gpu = torch.tensor(colors, device='cuda')
    triangle_vertices = vertices_gpu[torch.tensor(triangles, dtype=torch.long, device='cuda')]
    triangle_colors = colors_gpu[torch.tensor(triangles, dtype=torch.long, device='cuda')]
    triangle_data = torch.cat([triangle_vertices, triangle_colors], dim=2)
    return triangle_data


def generate_point_cloud(triangle_data, points_per_triangle, batch_size=4):
    triangle_data = triangle_data.to('cuda')
    a = triangle_data[:, 0, :3]  # 顶点1的坐标
    b = triangle_data[:, 1, :3]  # 顶点2的坐标
    c = triangle_data[:, 2, :3]  # 顶点3的坐标
    a_cpu = a.cpu()  # 顶点1的坐标
    b_cpu = b.cpu()  # 顶点2的坐标
    c_cpu = c.cpu()  # 顶点3的坐标
    color_a = triangle_data[:, 0, 3:]  # 顶点1的颜色
    color_a_cpu = color_a.cpu()  # 顶点1的颜色
    ab = b - a
    ac = c - a
    area = torch.norm(torch.cross(ab, ac), dim=1) / 2.0
    total_points = (area * points_per_triangle).long()
    # total_points[total_points < 1] = 1
    point_cloud = []
    point_colors = []
    for batch_start in range(0, triangle_data.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, triangle_data.shape[0])
        for i in range(batch_start, batch_end):
            n_points = total_points[i].item()
            if n_points > 1:
                r1 = torch.sqrt(torch.rand(n_points, 1, device='cuda'))
                r2 = torch.rand(n_points, 1, device='cuda')
                torch.cuda.empty_cache()
                random_points = ((1 - r1) * a[i] + r1 * ((1 - r2) * b[i] + r2 * c[i])).cpu()
                random_colors = color_a[i].unsqueeze(0).repeat(n_points, 1).cpu()
                torch.cuda.empty_cache()
                point_cloud.append(random_points)
                point_colors.append(random_colors)
            else:
                center_points = (a_cpu[i]+b_cpu[i]+c_cpu[i]).unsqueeze(0)/3
                point_cloud.append(center_points)
                random_colors = color_a_cpu[i].unsqueeze(0).repeat(1, 1)
                point_colors.append(random_colors)
    point_cloud = torch.cat(point_cloud, dim=0)
    point_colors = torch.cat(point_colors, dim=0)
    return point_cloud, point_colors


json_files = [file for file in os.listdir(base_dir + '3D-FRONT/') if file.endswith('.json')]
json_files_len = len(json_files)


def process_file(json_file, cnt, begin_idx):
    if not json_file.endswith('.json'):
        json_file += '.json'
    torch.set_grad_enabled(False)
    if cnt+begin_idx in [201, 209, 287, 483, 6570, 2706, 3551, 3821, 4000]: return
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    print('------ ', cnt + begin_idx, '/', json_files_len, '   1 ')
    with open(base_dir + '3D-FRONT/' + json_file, 'r') as f:
        scene_data = json.load(f)
        instance_table, room_Id = __create_instance_table(scene_data)
        print('------ ', cnt + begin_idx, '/', json_files_len, '   2 ')
        my_mesh = __create_mesh_table(scene_data)
        all_meshes = join(my_mesh, instance_table, 'id', 'ref', 'instance_')
        print('------ ', cnt + begin_idx, '/', json_files_len, '   3 ')
        furniture_table = __create_furniture_table(scene_data)
        if len(furniture_table['id']) <= 4:
            return
        furniture_all = join(furniture_table, instance_table, 'id', 'ref', 'instance_')
        all_meshes = create_triangle_meshes(all_meshes, room_Id)
        print('------ ', cnt + begin_idx, '/', json_files_len, '   4 ')
        Trans_Ms = []
        for furniture in zip(furniture_all['jid'], furniture_all['position'], furniture_all['rotation'], furniture_all['scale']):
            Trans_M = get_furniture_position(furniture, lambda jid: (
                os.path.join(shapeLocalSource, jid, 'raw_model.obj') if os.path.exists(
                    os.path.join(shapeLocalSource, jid, 'raw_model.obj')) else None,
            ))
            if Trans_M is not None:
                Trans_Ms.append(Trans_M)  # 将有效位置添加到列表中
        Trans_Ms = np.array(Trans_Ms)
        print('------ ', cnt + begin_idx, '/', json_files_len, '   5 ')
        obj_meshes = trans_obj(furniture_all['title'], furniture_all['jid'], furniture_all['room_Id'], room_Id, Trans_Ms)
        print('------ ', cnt + begin_idx, '/', json_files_len, '   6 ')
        if all_meshes == None or obj_meshes == None:
            return
        merge_mesh = merge_triangle_meshes(all_meshes + obj_meshes)
        # merge_mesh = merge_mesh.remove_degenerate_triangles()
        # merge_mesh = merge_mesh.remove_duplicated_triangles()
        # merge_mesh = merge_mesh.remove_duplicated_vertices()
        # merge_mesh.remove_non_manifold_edges()
        print('------ ', cnt + begin_idx, '/', json_files_len, '   7 ')

        with open('/Path/3D-FRONT-PC/3D-FRONT-camera/' + 'camera_' + json_file, 'r', encoding='utf-8') as f:
            cameras = json.load(f)
        print('smapling from mesh_o3d:', cnt+begin_idx, json_file)
        xyz, label, flag = _get_pcd_from_mesh_o3d(merge_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(merge_mesh))
        print('------ ', cnt + begin_idx, '/', json_files_len, '   8 ')
        if flag == None:
            print('------', cnt+begin_idx, '/', json_files_len, '  Mesh is not watertight')
            return
        pcd_group = group_point_clouds_by_roomid(np.concatenate((xyz, label), axis=1))
        scene_nojson = json_file.split('.')[0]
        print('------ ', cnt + begin_idx, '/', json_files_len, '   9 ')
        for i, camera in enumerate(cameras):
            if scene_nojson + '_' + str(i) + '.npy' not in scene_cameras: #修改了
                continue
            print('9', i, '--', scene_nojson + '_' + str(i))
            input_pts, input_room_Id, near, far = get_input(scene, camera, merge_mesh)
            if input_room_Id == None:
                continue
            count = np.sum(input_pts[:, 2] < 1.0)  # 计算 z 值小于 1 的数量
            if count > 80:
                continue
            filtered_points = pcd_group[input_room_Id]
            xyz_camera = world_to_camera_coordinates(filtered_points[:, :3], camera)
            # xyz_camera = filtered_points[:, :3]
            pcd_gt = np.concatenate((xyz_camera, filtered_points[:, 3:]), axis=1)
            pcd_gt = get_gt_in_camera_view(pcd_gt, (near, far), camera['fov'])
            if pcd_gt.shape[0] < 8192*2:
                continue
            labels = pcd_gt[:, 3]  # 取出标签列
            unique_labels = np.unique(labels)  # 找出所有唯一标签
            if len(unique_labels) <= 3:
                continue
            pcd_gt = pcd_downsample(pcd_gt, 8192)

            input_pts[:, :3] /= 6 # 3.4641
            pcd_gt[:, :3]    /= 6 # 3.4641
            np.save(base_dir + '3D-FRONT-depth_new/' + scene_nojson + '_' + str(i), input_pts[:, :3])
            np.save(base_dir +  '3D-FRONT-test_new/' + scene_nojson + '_' + str(i), np.ascontiguousarray(pcd_gt[:, :4]))
        with open(base_dir + 'data.txt', 'a') as file:
            file.write(str(cnt+begin_idx) + '----' + scene_nojson + '\n')
        print('------ ', cnt+begin_idx, '/', json_files_len, '   end ')

begin_idx = 1250



with open(base_dir + 'filtered.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
scene_list = []
scene_cameras = []
for line in lines:
    line = line.strip()  # 去除行末的换行符
    if '_' in line:
        parts = line.split('_', 1)  # 只分割一次
        scene_list.append(parts[0])
        scene_cameras.append(line)
scene_set = list(OrderedDict.fromkeys(scene_list))
json_files_len = len(scene_set)

# for i, filter_scene in enumerate(scene_set):
#     print('filter_scene', filter_scene)
#     process_file(filter_scene, i, 0)
parallel_results = Parallel(n_jobs=30, verbose=10, batch_size=1, timeout=20000)(
   delayed(process_file)(filter_scene, i, begin_idx) for i, filter_scene in enumerate(scene_set[begin_idx:])
)
assert isinstance(parallel_results, Iterable)


# process_file('00ad8345-45e0-45b3-867d-4a3c88c2517a.json', -1, begin_idx)


# for i, json_file in enumerate(json_files[begin_idx:]):
#     process_file(json_file, i, begin_idx)

# parallel_results = Parallel(n_jobs=20, verbose=10, batch_size=1, timeout=20000)(
#    delayed(process_file)(file, i, begin_idx) for i, file in enumerate(json_files[begin_idx:])
# )
# assert isinstance(parallel_results, Iterable)

# type_set = set()
# for i, json_file in enumerate(json_files):
#     with open(base_dir + '3D-FRONT/' + json_file, 'r') as f:
#         print('get type:', i, '/', json_files_len)
#         scene_data = json.load(f)
#         furniture_table = __create_furniture_table(scene_data)
#         titles = furniture_table['title']
#         for title in titles:
#             type_set.add(title)
#     print('type_set', len(type_set), type_set)
# type_set = set()
# for i, json_file in enumerate(json_files):
#     with open(base_dir + '3D-FRONT/' + json_file, 'r') as f:
#         print('get type:', i, '/', json_files_len)
#         scene_data = json.load(f)
#         mesh_table = __create_mesh_table(scene_data)
#         titles = mesh_table['type']
#         for title in titles:
#             type_set.add(title)
#     print('type_set', len(type_set), type_set)
FUNITURE_TYPES = ['plants', 'Two-seat Sofa', 'electronics', 'Wardrobe', 'accessory', 'Nightstand', 'stair',
                  'Sideboard / Side Cabinet / Console', 'Classic Chinese Chair', 'Children Cabinet', 'lighting',
                  'Dressing Table', 'sofa', 'kitchen cabinet', '200 - on the floor', 'barn door', 'unknown',
                  'Double Bed', 'mirror', 'shower', 'window', 'media unit', 'door', 'Lounge Chair / Book-chair / Computer Chair',
                  'Cabinet/Shelf/Desk', 'curtain', 'Lazy Sofa', 'Single bed', 'Sofa', 'Ceiling Lamp', 'Barstool',
                  'Dressing Chair', 'sink', 'Table', 'Bookcase / jewelry Armoire', 'vanity', 'shelf', 'armchair',
                  'Chair', 'Chaise Longue Sofa', 'Corner/Side Table', 'art', 'Bed Frame', 'Unknow', 'Lighting',
                  'Tea Table', 'toilet', 'Shelf', 'chair', 'Round End Table', 'Pier/Stool', '500 - attach to ceiling',
                  'Drawer Chest / Corner cabinet', 'attachment', 'rug', 'electric', 'water', 'Bunk Bed',
                  'Footstool / Sofastool / Bed End Stool / Stool', 'wardrobe', 'TV Stand', 'appliance', 'bath',
                  'Pendant Lamp', 'Desk', 'room', 'Kids Bed', 'smart home', 'Three-Seat / Multi-person sofa',
                  '400 - attach to wall', 'outdoor furniture', 'Others', 'Dining Table', 'build element',
                  'cabinet', 'material', 'Dining Chair', 'meter', 'recreation', 'storage unit', 'table', 'bed',
                  'Logo', 'basin', 'L-shaped Sofa', '300 - on top of others', 'Bed', 'Wine Cooler']
# categories_in_3d = {item['category'].lower() for item in _FURNITURE_CATEGORIES_3D}
# not_appeared = [f for f in FUNITURE_TYPES if f.lower() not in categories_in_3d]
# super_categories_3d = {item['category'].lower() for item in _SUPER_CATEGORIES_3D}
# not_appeared = [
#     f['super-category'] for f in _FURNITURE_CATEGORIES_3D
#     if all(f['super-category'].lower() not in cat.lower() for cat in super_categories_3d)
# ]
# print('not_appeared', not_appeared)





