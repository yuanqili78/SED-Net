color_dict = [[255, 174, 162],[255,126,195],[199,163,242],[53,198,197],[139,211,148],[169,185,241],[174,84,201],
              [210,204,124],[213,189,179],[116,128,176],[139,232,255],[227,252,186],[141,141,141],[46,117,182],
              [154,117,0],[136,127,249],[142,186,234],[249,243,155],[158,179,124],[173,148,146],[204,154,116],
              [243,132,77],[208,237,83],[187,243,227],[90,156,228],[103,69,211],[146,99,181],[221,83,106],
              [115,23,38],[235,194,111]]

circle_vnum = 49
min_delta = 5e-2
doublePI = 6.283185307179586

import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import triangle as tr

def get_paraPoint(para, edge_con):
    s_type = para[0]

    ret = []
    ret_normal = []
    ret_v = []
    ret_f = []
    n_points = 5000

    # pre-processing
    for idx in edge_con:
        edge = edge_con[idx]
        if edge[0] == "circle" and (len(edge[1]) != 3):
            edge_con[idx][1] = edge[1][0]

    if s_type == "plane":
        #axis = np.array(para[1].cpu())
        axis = para[1]
        z = np.array([axis[0], axis[1], axis[2]])
        #d = np.array(para[2].cpu())
        d = para[2]
        l = z * d
        # print(l)
        x, y = find_vertical_vector(z)

        # quick two circle mesh
        v_min, v_max = 10, -10
        for idx in edge_con:
            edge = edge_con[idx]
            # print(edge)
            if edge[0] == "circle":
                v_tmp = np.dot(edge[1], z)
                # print(edge, v_tmp)
                if v_tmp < v_min:
                    v_min = v_tmp
                    bottom = edge
                if v_tmp > v_max:
                    v_max = v_tmp
                    top = edge
        # print(edge_con)
        if len(edge_con) == 2 and next(iter(edge_con.items()))[1][0] != 'line':
            for i, idx in enumerate(edge_con):
                if i == 0:
                    bottom = edge_con[idx]
                else:
                    top = edge_con[idx]
            #bottom = edge_con[idx]
            #bottom = edge_con[0]
            #top = edge_con[1]
            # print(bottom)
            # print(top)
            ret_v, ret_f = doubleCircleEdge_mesh(bottom, top)
            return ret_v, ret_f

        line_edge, circle_edge = [], []
        for idx in edge_con:
            edge = edge_con[idx]
            if edge[0] == "line":
                line_edge.append(edge)
            elif edge[0] == "circle":
                circle_edge.append(edge)

        # print(edge_con)
        # print(line_edge)
        vertices = []
        segments, segments_len = [], 0
        holes = []
        circle2line_edge = []
        for i in range(len(circle_edge)):
            edge = circle_edge[i]
            # if abs(edge[6] - edge[5] - 2 * np.pi) > 1e-3:  #incomplete circle
            if True :
                circle_vertices_absCoord = sample_circleEdge_absCoord(edge)
                # print(len(circle_vertices_absCoord))

                for j in range(len(circle_vertices_absCoord) - 1):
                    new_line = two_point_paraLine(circle_vertices_absCoord[j], circle_vertices_absCoord[j + 1])
                    circle2line_edge.append(new_line)

        line_edge.extend(circle2line_edge)
        
        polygon_vertex_set, polygon_area, max_area = get_polygon_set(l, x, y, line_edge)
        
        for i in range(len(polygon_vertex_set)):
            vertices.extend(polygon_vertex_set[i])
            segments.extend(seq_connect(segments_len, segments_len + len(polygon_vertex_set[i]) - 1))
            # segments.extend(seq_connect(0, 30))
            segments_len += len(polygon_vertex_set[i])
            if len(polygon_vertex_set[i]) > 2 and polygon_area[i] < max_area - 1e-3:
                holes.append(polygon_center(polygon_vertex_set[i]))
        
        if holes != []:
            triange_input = {'vertices': vertices, "segments": segments, "holes": holes}
        else:
            triange_input = {'vertices': vertices, "segments": segments}
        t1 = tr.triangulate(triange_input, 'p')
        tr.compare(plt, triange_input, t1)
        
        ret_v, ret_f = triangleForm_to_OBJ(t1, l, x, y)
        
        return ret_v, ret_f

    if s_type == "cylinder":
        l = np.array(para[2])
        r = para[3]
        axis = np.array(para[1])
        
        z = np.array([axis[0], axis[1], axis[2]])
        x, y = find_vertical_vector(z)

        v_min, v_max = 10, -10
        u_min, u_max = 0, 2 * np.pi
        for idx in edge_con:
            edge = edge_con[idx]
            # print(edge)
            if edge[0] == "circle":
                v_tmp = np.dot(edge[1], z)
                # print(edge, v_tmp)
                if edge[5] > u_min:
                    u_min = edge[5]
                if edge[6] < u_max:
                    u_max = edge[6]
                if v_tmp < v_min:
                    v_min = v_tmp
                    bottom = edge
                if v_tmp > v_max:
                    v_max = v_tmp
                    top = edge
        #print(bottom)
        #print(top)
        ret_v, ret_f = doubleCircleEdge_mesh(bottom, top)
        return ret_v, ret_f

    if s_type == "cone":
        # print("---cone---")
        apex = np.array(para[1])  # 
        axis = np.array(para[2])
        z = np.array([axis[0], axis[1], axis[2]])
        x, y = find_vertical_vector(z)
        a = np.array(para[3])  # 

        l = apex
        # bottom
 
        for idx in edge_con:
            edge = edge_con[idx]

            if edge[0] == "circle":
                circle_c = edge[1]
  
                if np.linalg.norm(circle_c - apex) > np.linalg.norm(l - apex):
                    l = circle_c
                    bottom = edge
        r = np.linalg.norm(l - apex) * np.tan(a)
        # print(z)

        v_min, v_max = 0, 0
        u_min, u_max = 0, 2 * np.pi

        # print(len(edge_con))
        if len(edge_con) == 1:
            ret_v, ret_f = pointCircleEdge_mesh(apex, bottom)
            return ret_v, ret_f

        for idx in edge_con:
            edge = edge_con[idx]
            # print(edge)
            if edge[0] == "circle":
                circle_c = edge[1]
                v_tmp = np.linalg.norm(circle_c - l)
                # print("###", circle_c, l, v_tmp, v_min)
                if v_tmp > v_min:
                    v_min = v_tmp
                    top = edge
        # print(bottom)
        ret_v, ret_f = doubleCircleEdge_mesh(bottom, top)
        return ret_v, ret_f

    if s_type == "sphere":
        sphere_c = np.array(para[1])
        sphere_r = np.array(para[2])
        
        v_min, v_max = 0, 2 * np.pi
        u_min, u_max = 0, 2 * np.pi
        for idx in edge_con:
            edge = edge_con[idx]
            # print(edge)

            if edge[0] == "circle":
                circle_c = edge[1]
                circle_r = edge[4]
                x = edge[2]
                y = edge[3]
                z = np.cross(x, y) / np.linalg.norm(np.cross(x, y))
                # print(circle_r, sphere_r)
                v_tmp = np.arccos(circle_r / sphere_r)
                

        v_min, v_max = np.pi + v_tmp, 2 * np.pi - v_tmp
        #print(-v_tmp, v_tmp)
        ret_v, ret_f = sphere_mesh(sphere_c, sphere_r, x, y, z, v_min, v_max)
        return ret_v, ret_f

    # ret = np.array(ret)
    # ret_normal = np.array(ret_normal)
    return ret, ret_normal

def find_vertical_vector(base):
    eps = 1e-4
    if abs(base[0]) > eps:
        ret_x = [-base[1] / base[0], 1, 0]
    elif abs(base[1]) > eps:
        ret_x = [0, -base[2] / base[1], 1]
    elif abs(base[2]) > eps:
        ret_x = [1, 0, -base[0] / base[2]]

    ret_y = np.cross(base, ret_x)
    ret_x = ret_x / np.linalg.norm(ret_x)
    ret_y = ret_y / np.linalg.norm(ret_y)
    return ret_x, ret_y

def get_polygon_set(l, x, y, line_edge):
    # print(line_edge)
    # print('\n')
    vertex_set = []
    ret_area = []
    for edge in line_edge:
        # print(edge)
        line_d = edge[1]
        line_l = edge[2]
        line_range = edge[3]
        # print(line_range)
        pt0 = line_l + line_range[0] * line_d
        pt1 = line_l + line_range[1] * line_d

        # print(edge, pt0, pt1)
        paraCoord0 = getParaCoord(pt0, l, x, y)
        paraCoord1 = getParaCoord(pt1, l, x, y)
            # print("***", pt0, l + paraCoord0[0] * x + paraCoord0[1] * y)
            # print("***", pt1, l + paraCoord1[0] * x + paraCoord1[1] * y)
        flag = True
        for i in range(len(vertex_set)):
            if abs(vertex_set[i][0] - paraCoord0[0]) + abs(vertex_set[i][1] - paraCoord0[1]) < 1e-3:
                flag = False
                break
        if flag:
            vertex_set.append((paraCoord0[0], paraCoord0[1]))
        flag = True
        for i in range(len(vertex_set)):
            if abs(vertex_set[i][0] - paraCoord1[0]) + abs(vertex_set[i][1] - paraCoord1[1]) < 1e-3:
                flag = False
                break
        if flag:
            vertex_set.append((paraCoord1[0], paraCoord1[1]))

    # vertex_set = Clockwise_sort(vertex_set)
    sorted_sets = []
    max_area = 0
    choosen = [False] * len(vertex_set)
    # print(len(vertex_set))
    for i in range(len(vertex_set)):
        if choosen[i]:
            continue
        sorted_set = []
        sorted_set.append(vertex_set[i])
        choosen[i] = True
        tmp = vertex_set[i]
        # another = find_another_point(tmp, l, x, y, line_edge, vertex_set, choosen)
        another = first_find_another_point(tmp, l, x, y, line_edge, vertex_set, choosen)
        if len(another) == 1:
            another = another[0]
            # choosen[find_index(vertex_set, another)] = True
            while not CoordApprox(another, (-100, -100)) and not CoordApprox(another, tmp):
                index = find_index(vertex_set, another)
                choosen[index] = True
                sorted_set.append(vertex_set[index])
                tmp = another
                another = find_another_point(tmp, l, x, y, line_edge, vertex_set, choosen)
                # if len(vertex_set) == 5:
                #    print(another)
                # print(while_count)
                if CoordApprox(another, (-100, -100)):
                    break
        else:
            a1, a2 = another[0], another[1]
            # choosen[find_index(vertex_set, a1)] = True
            # choosen[find_index(vertex_set, a2)] = True
            # print(another, a1, a2)
            sorted_set1, sorted_set2 = [], []
            while not CoordApprox(a1, (-100, -100)) and not CoordApprox(a1, tmp):
                index = find_index(vertex_set, a1)
                choosen[index] = True
                sorted_set1.append(vertex_set[index])
                tmp = a1
                a1 = find_another_point(tmp, l, x, y, line_edge, vertex_set, choosen)
                if CoordApprox(a1, (-100, -100)):
                    break
            while not CoordApprox(a2, (-100, -100)) and not CoordApprox(a2, tmp):
                index = find_index(vertex_set, a2)
                choosen[index] = True
                sorted_set2.append(vertex_set[index])
                tmp = a2
                a2 = find_another_point(tmp, l, x, y, line_edge, vertex_set, choosen)
                if CoordApprox(a2, (-100, -100)):
                    break
            sorted_set = []
            sorted_set.extend(reversed(sorted_set2))
            sorted_set.append(vertex_set[i])
            sorted_set.extend(sorted_set1)
        # print(sorted_set)
        sorted_sets.append(sorted_set)
        now_area = polygon_area(sorted_set)
        ret_area.append(now_area)
        if now_area > max_area:
            max_area = now_area
    return sorted_sets, ret_area, max_area

def triangleForm_to_OBJ(triangle, l, x, y):
    ret_v, ret_f = [], []
    for i in range(len(triangle["vertices"])):
        para = triangle["vertices"][i]
        ret_v.append(l + para[0] * x + para[1] * y)
    for i in range(len(triangle["triangles"])):
        tmp = triangle["triangles"][i]
        tmp[0] += 1
        tmp[1] += 1
        tmp[2] += 1
        ret_f.append(tmp)
    return ret_v, ret_f

def sample_circleEdge_absCoord(edge):
    # print(edge)
    ret_v = []
    # if len(circle_edge) == 1:
    x, y = edge[2], edge[3]
    l = edge[1]
    r = edge[4]
    t_min, t_max = edge[5], edge[6]
    # if t_max - 2 * np.pi >= 0:
    #    t_max += 2 * (t_max - t_min) / circle_vnum
    t_interval = 2 * np.pi / circle_vnum
    tmp_circle_vnum = int((t_max - t_min) / (2 * np.pi) * circle_vnum) + 1
    for i in range(tmp_circle_vnum):
        t = t_min + i * t_interval
        v_now = l + r * np.cos(t) * x + r * np.sin(t) * y
        ret_v.append(v_now)
        t += t_interval
    if abs(t_max - t_min - 2 * np.pi) > 1e-3:
        v_now = l + r * np.cos(t_max) * x + r * np.sin(t_max) * y
        ret_v.append(v_now)
    #print(len(ret_v))
    return ret_v

def doubleCircleEdge_mesh(circle1, circle2):
    ret_v, ret_f = [], []
    # print(circle1[2],circle2[2])
    # print(circle1[3], circle2[3])
    circle2[2], circle2[3] = circle1[2], circle1[3]  # axis
    ret_v1 = sample_circleEdge_absCoord(circle1)
    ret_v2 = sample_circleEdge_absCoord(circle2)
    #print(ret_v1[:3], ret_v2[:3])
    t_range = min(circle1[6] - circle1[5], circle2[6] - circle2[5])
    if abs(t_range - 2 * np.pi) > 1e-3:
        closed = False
    else:
        closed = True
    """
    ret_v.pop(circle_vnum + 1)  #去掉圆心
    ret_v.pop(0)
    """
    tmp_vnum = len(ret_v1)
    ret_v.extend(ret_v1)
    ret_v.extend(ret_v2)
    for i in range(1, tmp_vnum + 1):
        if i < tmp_vnum:
            face_tmp1 = [i, tmp_vnum + i, tmp_vnum + i + 1]
            face_tmp2 = [i, i + 1, tmp_vnum + i + 1]
        elif closed:
            face_tmp1 = [i, tmp_vnum + i, tmp_vnum + 1]
            face_tmp2 = [i, 1, tmp_vnum + 1]
        else:
            face_tmp1 = [tmp_vnum - 1, tmp_vnum, 2 * tmp_vnum]
            face_tmp2 = [2 * tmp_vnum, 2 * tmp_vnum - 1, tmp_vnum - 1]
            # print(i, tmp_vnum - 1, tmp_vnum, 2 * tmp_vnum, 2 * tmp_vnum - 1)
        ret_f.append(face_tmp1)
        ret_f.append(face_tmp2)
    # ret_f.append([tmp_vnum - 1, tmp_vnum, 2 * tmp_vnum, 2 * tmp_vnum - 1])
    return ret_v, ret_f

def sphere_mesh(l, r, x, y, z, v_min, v_max):
    ret_v, ret_f = [], []
    v_interval = 0.05
    u_min, u_max = 0, 2 * np.pi
    u_interval = (u_max - u_min) / circle_vnum
    v_num = 0
    ret_v0, ret_v1 = [], []
    u, v = u_min, v_min
    while v <= v_max + v_interval:
        u = u_min
        if ret_v0 == []:
            while u <= u_max:
                p_now = l + r * np.cos(v) * (np.cos(u) * x + np.sin(u) * y) + r * np.sin(v) * z
                ret_v0.append(p_now)
                u += u_interval
            ret_v.extend(ret_v0)
        else:
            while u <= u_max:
                p_now = l + r * np.cos(v) * (np.cos(u) * x + np.sin(u) * y) + r * np.sin(v) * z
                ret_v1.append(p_now)
                u += u_interval

            for i in range(1, circle_vnum + 1):
                if i < circle_vnum:
                    face_tmp1 = [v_num + i, v_num + circle_vnum + i, v_num + circle_vnum + i + 1]
                    face_tmp2 = [v_num + i, v_num + i + 1, v_num + circle_vnum + i + 1]
                else:
                    face_tmp1 = [v_num + i, v_num + circle_vnum + i, v_num + circle_vnum + 1]
                    face_tmp2 = [v_num + i, v_num + 1, v_num + circle_vnum + 1]
                ret_f.append(face_tmp1)
                ret_f.append(face_tmp2)

            v_num += circle_vnum
            ret_v.extend(ret_v1)
            ret_v0 = ret_v1.copy()
            ret_v1 = []
        v += v_interval
    return ret_v, ret_f

def two_point_paraLine(p1, p2):
    p1, p2 = np.array(p1), np.array(p2)
    d = p2 - p1
    d = d / np.linalg.norm(d)
    q = - np.dot(p1, d) / np.linalg.norm(d) ** 2
    l = p1 + q * d
    # print(l, d, p1, p2)
    if abs(d[0]) > 1e-2:
        d1 = (p1[0] - l[0]) / d[0]
        d2 = (p2[0] - l[0]) / d[0]
        # print("N1", l + d1 * d, l + d2 * d)
    elif abs(d[1]) > 1e-2:
        d1 = (p1[1] - l[1]) / d[1]
        d2 = (p2[1] - l[1]) / d[1]
    else:
        d1 = (p1[2] - l[2]) / d[2]
        d2 = (p2[2] - l[2]) / d[2]
    d_min = min(d1, d2)
    d_max = max(d1, d2)
    # print(d1, d2)
    ret = ["line", d, l, [d_min, d_max]]
    # print("newline:", ret)
    return ret

def seq_connect(start, end):
    ret = []
    for i in range(start, end):
        ret.append([i, i + 1])
    ret.append([end, start])
    return ret

def polygon_center(polygon_vertices):
    center_x, center_y = 0, 0
    for i in range(len(polygon_vertices)):
        center_x += polygon_vertices[i][0]
        center_y += polygon_vertices[i][1]
    center_x /= len(polygon_vertices)
    center_y /= len(polygon_vertices)
    return [center_x, center_y]

def pointCircleEdge_mesh(point, circle):
    ret_v, ret_f = [], []
    # ret_v, _ = pureCircleEdge_mesh([circle1, circle2])
    # print(circle1[2],circle2[2])
    # print(circle1[3], circle2[3])
    ret_v1 = sample_circleEdge_absCoord(circle)
    # print(ret_v1[:3], ret_v2[:3])
    t_range = circle[6] - circle[5]
    if abs(t_range - 2 * np.pi) > 1e-3:
        closed = False
    else:
        closed = True

    tmp_vnum = len(ret_v1)
    ret_v.extend(ret_v1)
    ret_v.append(point)
    for i in range(1, tmp_vnum + 1):
        if i < tmp_vnum:
            face_tmp = [i, i + 1, tmp_vnum + 1]
        elif closed:
            face_tmp = [i, 1, tmp_vnum + 1]

        ret_f.append(face_tmp)
    #for i in range(len(ret_v)):
    #    ret_v[i] = ret_v[i].tolist()
    #print(point)
    #print(ret_f)
    return ret_v, ret_f

def getParaCoord(pt, l, x, y):
    if abs(x[1] * y[0] - x[0] * y[1]) > min_delta:
        u0 = (pt[1] * y[0] - pt[0] * y[1] + l[0] * y[1] - l[1] * y[0]) / (x[1] * y[0] - x[0] * y[1])
        v0 = (pt[1] * x[0] - pt[0] * x[1] + l[0] * x[1] - l[1] * x[0]) / (y[1] * x[0] - y[0] * x[1])
    else:
        u0, v0 = 10, 10

    if abs(x[2] * y[0] - x[0] * y[2]) > min_delta:
        u1 = (pt[2] * y[0] - pt[0] * y[2] + l[0] * y[2] - l[2] * y[0]) / (x[2] * y[0] - x[0] * y[2])
        v1 = (pt[2] * x[0] - pt[0] * x[2] + l[0] * x[2] - l[2] * x[0]) / (y[2] * x[0] - y[0] * x[2])
    else:
        u1, v1 = 100, 100

    if abs(x[2] * y[1] - x[1] * y[2]) > min_delta:
        u2 = (pt[2] * y[1] - pt[1] * y[2] + l[1] * y[2] - l[2] * y[1]) / (x[2] * y[1] - x[1] * y[2])
        v2 = (pt[2] * x[1] - pt[1] * x[2] + l[1] * x[2] - l[2] * x[1]) / (y[2] * x[1] - y[1] * x[2])
    else:
        u2, v2 = 1000, 1000

    
    """
    if not (-1 <= u0 <= 1) or not (-1 <= u1 <= 1) or not (-1 <= u2 <= 1):
        if (-1 <= u0 <= 1):
            u = u0
            v = v0
        if (-1 <= u1 <= 1):
            u = u1
            v = v1
        if (-1 <= u2 <= 1):
            u = u2
            v = v2
        return u, v
    """
    if not (-2 <= u0 <= 2) or not (-2 <= u1 <= 2) or not (-2 <= u2 <= 2):
        if (-2 <= u0 <= 2):
            u = u0
            v = v0
        if (-2 <= u1 <= 2):
            u = u1
            v = v1
        if (-2 <= u2 <= 2):
            u = u2
            v = v2
        return u, v


    #print("-----")
    if (abs(u0 - u1) < min_delta or abs(u0 - u2) < min_delta):
        u = u0
    elif abs(u1 - u2) < min_delta:
        u = u1
    else:
        print(pt, l, x, y)
        print("There is a problem!", u0, u1, u2)

    if abs(v0 - v1) < min_delta or abs(v0 - v2) < min_delta:
        v = v0
    elif abs(v1 - v2) < min_delta:
        v = v1
    else:
        #print(pt, l, x, y)
        print("There is a problem!", v0, v1, v2)

    return u, v

def first_find_another_point(coord, l, x, y, line_edge, v_set, choosen):
    # print(choosen)
    ret = []
    for edge in line_edge:
        line_d = edge[1]
        line_l = edge[2]
        line_range = edge[3]
        pt0 = line_l + line_range[0] * line_d
        pt1 = line_l + line_range[1] * line_d
        paraCoord0 = getParaCoord(pt0, l, x, y)
        paraCoord1 = getParaCoord(pt1, l, x, y)
        if CoordApprox(paraCoord0, coord) and not choosen[find_index(v_set, paraCoord1)]:
            ret.append(paraCoord1)
        if CoordApprox(paraCoord1, coord) and not choosen[find_index(v_set, paraCoord0)]:
            ret.append(paraCoord0)
        
    if len(ret) > 0:
        return ret
    else:
        return [(-100, -100)]

def CoordApprox(coord1, coord2):
    # print(coord1, coord2)
    if abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]) < 1e-3:
        return True
    else:
        return False

def find_index(v_set, coord):
    # print(coord)
    for i in range(len(v_set)):
        if CoordApprox(v_set[i], coord):
            return i

def find_another_point(coord, l, x, y, line_edge, v_set, choosen):
    # print(choosen)
    for edge in line_edge:
        line_d = edge[1]
        line_l = edge[2]
        line_range = edge[3]
        pt0 = line_l + line_range[0] * line_d
        pt1 = line_l + line_range[1] * line_d
        paraCoord0 = getParaCoord(pt0, l, x, y)
        paraCoord1 = getParaCoord(pt1, l, x, y)
        if CoordApprox(paraCoord0, coord) and not choosen[find_index(v_set, paraCoord1)]:
            return paraCoord1
        if CoordApprox(paraCoord1, coord) and not choosen[find_index(v_set, paraCoord0)]:
            return paraCoord0
        
    return (-100, -100)

def polygon_area(polygon_vertices):
    ret = 0
    for i in range(len(polygon_vertices)):
        if i != len(polygon_vertices) - 1:
            sub_item = abs(
                polygon_vertices[i][0] * polygon_vertices[i + 1][1] - polygon_vertices[i + 1][0] * polygon_vertices[i][
                    1])
        else:
            sub_item = abs(
                polygon_vertices[i][0] * polygon_vertices[0][1] - polygon_vertices[0][0] * polygon_vertices[i][1])
        ret += sub_item
    return ret / 2

def save_obj(filename, v_list, f_list):
    with open(filename, "w") as f:
        for vertex in v_list:
            if len(vertex) == 3:
                f.write('v ' + str(vertex[0]) + " " + str(vertex[1]) + " " + str(vertex[2]))
            elif len(vertex) == 4:
                #print(vertex)
                f.write('v ' + str(vertex[0][0]) + " " + str(vertex[0][1]) + " " + str(vertex[0][2]))
                f.write(' ' + str(vertex[1]) + " " + str(vertex[2]) + " " + str(vertex[3]))
            else:
                #print(vertex)
                f.write('v ' + str(vertex[0]) + " " + str(vertex[1]) + " " + str(vertex[2]))
                f.write(' ' + str(vertex[3]) + " " + str(vertex[4]) + " " + str(vertex[5]))
            f.write('\n')

        f.write("\n")

        for face in f_list:
            if len(face) == 3:
                f.write('f ' + str(face[0]) + " " + str(face[1]) + " " + str(face[2]))
            else:
                f.write('f ' + str(face[0]) + " " + str(face[1]) + " " + str(face[2]) + " " + str(face[3]))
            f.write('\n')

def add_index(face_set, num):
    ret = face_set.copy()
    for i in range(len(face_set)):
        ret[i][0] += num
        ret[i][1] += num
        ret[i][2] += num
        if len(ret[i]) > 3:
            ret[i][3] += num
    return ret

def read_data(param,inter_lines):
    ret_primitive_ids = []
    ret_fitter = []
    ret_inter_para_set = {}
    with open(param,'r') as file:
        for line in file:
            parts = line.split(':')
            ret_primitive_ids.append(parts[0].split(' ')[1])
            parts = parts[1].split(',')
            # print(parts)
            shape = parts[0].strip()
            if shape == 'plane':
                orientation = array([float(x) for x in parts[1].strip()[1:-1].split()])
                distance = float(parts[2].strip()[1:-1])
                ret_fitter.append([shape,orientation,distance])
            if shape == 'cylinder':
                center_point = array([float(x) for x in parts[1].strip()[1:-1].split()])
                orientation = array([float(x) for x in parts[2].strip()[1:-1].split()])
                radius = float(parts[3].strip())
                ret_fitter.append([shape,center_point,orientation,radius])
            if shape == 'cone':
                vertax = array([float(x) for x in parts[1].strip()[1:-1].split()])
                orientation = array([float(x) for x in parts[2].strip()[1:-1].split()])
                angle = float(parts[3].strip()[1:-1])
                ret_fitter.append([shape,vertax,orientation,angle])
            if shape == 'sphere':
                center_point = array([float(x) for x in parts[1].strip()[1:-1].split()])
                radius = float(parts[2].strip())
                ret_fitter.append([shape,center_point,radius])
            if shape == 'none':
                ret_fitter.append([shape])

    with open(inter_lines,'r') as file:
        
        ret_inter_para_set = eval(file.readline())
        for key in ret_inter_para_set:
            to_be_del = []
            for key2 in ret_inter_para_set[key]:
                if ret_inter_para_set[key][key2][0] == 'circle':
                    if len(ret_inter_para_set[key][key2]) == 5:
                        ret_inter_para_set[key][key2].append(0)
                        ret_inter_para_set[key][key2].append(doublePI)
                    else:
                        left=ret_inter_para_set[key][key2][5][0]
                        right=ret_inter_para_set[key][key2][5][1]
                        ret_inter_para_set[key][key2][5]=left
                        ret_inter_para_set[key][key2].append(right)
                    ret_inter_para_set[key][key2][1]=array(ret_inter_para_set[key][key2][1])
                    ret_inter_para_set[key][key2][2]=array(ret_inter_para_set[key][key2][2])
                    ret_inter_para_set[key][key2][3]=array(ret_inter_para_set[key][key2][3])
                if ret_inter_para_set[key][key2][0] == 'line':
                    if len(ret_inter_para_set[key][key2][3]) == 1:
                        to_be_del.append(key2)
                    ret_inter_para_set[key][key2][1]=array(ret_inter_para_set[key][key2][1])
                    ret_inter_para_set[key][key2][2]=array(ret_inter_para_set[key][key2][2])
                    ret_inter_para_set[key][key2][3]=array(ret_inter_para_set[key][key2][3])
            for key2 in to_be_del:
                del ret_inter_para_set[key][key2]

    
    return ret_primitive_ids,ret_fitter,ret_inter_para_set   


def arg2mesh(output_dir,param,inter_lines):
    primitive_ids, fitter, inter_para_set = read_data(param,inter_lines)
    all_vertexes = []
    all_faces = []
    all_color = []
    for i in range(len(fitter)):
        para1 = fitter[i]
        # print(para1)
        if para1[0] == "plane":
            inst_id_1 = primitive_ids[i]
            edge_condition = inter_para_set[inst_id_1]
            #print("***", i, inst_id_1)
            #print("***", edge_condition)
            v_tmp, f_tmp = get_paraPoint(para1, edge_condition)
            #print(v_tmp)
            color = np.array([color_dict[int(inst_id_1)]] * len(v_tmp))
            v_tmp = np.c_[v_tmp, color]
            #print(v_tmp.shape)
            #print(two_point_paraLine([0.23344488014481982, 0, 0], [0.23344488014481982, 1, 0]))
            save_obj(output_dir + str(inst_id_1) + "_plane.obj", v_tmp, f_tmp)
            # print(len(all_vertexes))
            f_tmp = add_index(f_tmp, len(all_vertexes))
            all_vertexes.extend(v_tmp)
            all_faces.extend(f_tmp)
            # all_color.extend(color)
        if para1[0] == "cylinder":
            inst_id_1 = primitive_ids[i]
            edge_condition = inter_para_set[inst_id_1]
            # print("***", para1)
            # print(edge_condition)
            v_tmp, f_tmp = get_paraPoint(para1, edge_condition)
            color = np.array([color_dict[int(inst_id_1)]] * len(v_tmp))
            v_tmp = np.c_[v_tmp, color]
            save_obj(output_dir + str(inst_id_1) + "_cylinder.obj", v_tmp, f_tmp)
            f_tmp = add_index(f_tmp, len(all_vertexes))
            all_vertexes.extend(v_tmp)
            all_faces.extend(f_tmp)
        if para1[0] == "cone":
            inst_id_1 = primitive_ids[i]
            edge_condition = inter_para_set[inst_id_1]
            # print("***", para1)
            v_tmp, f_tmp = get_paraPoint(para1, edge_condition)
            color = np.array([color_dict[int(inst_id_1)]] * len(v_tmp))
            #print(np.c_[v_tmp[0], color[0]])
            v_tmp = np.c_[v_tmp, color]
            #print(v_tmp[0])
            save_obj(output_dir + str(inst_id_1) + "_cone.obj", v_tmp, f_tmp)
            f_tmp = add_index(f_tmp, len(all_vertexes))
            all_vertexes.extend(v_tmp)
            all_faces.extend(f_tmp)
        if para1[0] == "sphere":
            inst_id_1 = primitive_ids[i]
            edge_condition = inter_para_set[inst_id_1]
            # print("***", para1)
            v_tmp, f_tmp = get_paraPoint(para1, edge_condition)
            color = np.array([color_dict[int(inst_id_1)]] * len(v_tmp))
            v_tmp = np.c_[v_tmp, color]
            save_obj(output_dir + str(inst_id_1) + "_sphere.obj", v_tmp, f_tmp)
            f_tmp = add_index(f_tmp, len(all_vertexes))
            all_vertexes.extend(v_tmp)
            all_faces.extend(f_tmp)
    

if __name__ == "__main__":
    arg2mesh("E:\\arg2mesh\\test\\output\\","E:\\arg2mesh\\test\\param_4147.txt","E:\\arg2mesh\\test\\param_inter_lines_4147.json")
