import taichi as ti
from argparse import ArgumentParser
import numpy as np
import os
import trimesh
import torch
import math
from pathlib import Path
import json
import tetgen
from common import *

ti.init(arch=ti.cpu)

# quaternion helper functions
@ti.func
def quat_mul(a, b)->ti.Vector:
    return ti.Vector([a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                      a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                      a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3],
                      a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1]])

@ti.func
def quat_mul_scalar(a, b)->ti.Vector:
    return ti.Vector([a[0] * b, a[1] * b, a[2] * b, a[3] * b])

@ti.func
def quat_add(a, b)->ti.Vector:
    return ti.Vector([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])

@ti.func
def quat_subtraction(a, b)->ti.Vector:
    return ti.Vector([a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]])

@ti.func
def quat_normal(a)->ti.f32:
    return ti.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3])

@ti.func
def quat_to_matrix(q)->ti.Matrix:
    q = q.normalized()
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ti.Matrix([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                      [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                      [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])

@ti.func
def quat_inverse(q)->ti.Vector:
    # the inverse of a quaternion is its conjugate divided by its norm
    norm_squared = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
    return ti.Vector([q[0], -q[1], -q[2], -q[3]]) / norm_squared

@ti.func
def Get_Cross_Matrix(a)->ti.Matrix:
    return ti.Matrix([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    # A = ti.Matrix.zero(dt=ti.f32, n=4, m=4)
    # A[0, 0] = 0
    # A[0, 1] = -a[2]
    # A[0, 2] = a[1]
    # A[1, 0] = a[2]
    # A[1, 1] = 0
    # A[1, 2] = -a[0]
    # A[2, 0] = -a[1]
    # A[2, 1] = a[0]
    # A[2, 2] = 0
    # A[3, 3] = 1
    # return A

# the euler angle is in degree, we first conver it to radian
def form_euler(euler_angle):
    # convert euler angle to quaternion
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    phi = math.radians(euler_angle[0] / 2)
    theta = math.radians(euler_angle[1] / 2)
    psi = math.radians(euler_angle[2] / 2)

    w = math.cos(phi) * math.cos(theta) * math.cos(psi) + math.sin(phi) * math.sin(theta) * math.sin(psi)
    x = math.sin(phi) * math.cos(theta) * math.cos(psi) - math.cos(phi) * math.sin(theta) * math.sin(psi)
    y = math.cos(phi) * math.sin(theta) * math.cos(psi) + math.sin(phi) * math.cos(theta) * math.sin(psi)
    z = math.cos(phi) * math.cos(theta) * math.sin(psi) - math.sin(phi) * math.sin(theta) * math.cos(psi)

    return [w, x, y, z]

@ti.func
def vec_to_quat(vec):
    return ti.Vector([0.0, vec[0], vec[1], vec[2]])

@ti.func
def quat_to_vec(quat):
    return ti.Vector([quat[1], quat[2], quat[3]])

@ti.func
def get_current_position(initial_vertex_position, quaternion, translation, initial_mass_center)->ti.Vector:
    # Step 1: Calculate initial offset
    initial_offset = initial_vertex_position - initial_mass_center

    # Step 2: Apply rotation using quaternion
    rotated_offset = quat_mul(quat_mul(quaternion, vec_to_quat(initial_offset)), quat_inverse(quaternion))

    # Step 3: Apply translation
    current_position = quat_to_vec(rotated_offset) + translation

    return current_position
    
@ti.data_oriented
class rigid_body:

    current_frame = 0
    T = ti.Matrix.field(4, 4, dtype=ti.f32, shape=1)
    T2 = ti.Matrix.field(4, 4, dtype=ti.f32, shape=1)
    frame_dt = 1.0 / 60.0
    substep = 30
    dt = frame_dt / substep

    def __init__(self, mesh_file_name, options):
        assert options is not None
        assert mesh_file_name is not None, 'mesh_file_name is None. You need to privide a mesh file name.'
            # floor
        floor_vertices =  np.array([[-5.0, 0.0, -5.0], [-5.0, 0.0, 5.0], [5.0, 0.0, 5.0], [5.0, 0.0, -5.0]])
        floor_faces = np.array([[0, 1, 2], [0, 2, 3]]).flatten()
        self.floor_vertices = ti.Vector.field(3, dtype=ti.f32, shape=4)
        self.floor_faces = ti.field(dtype=ti.i32, shape=6)
        self.floor_vertices.from_numpy(floor_vertices)
        self.floor_faces.from_numpy(floor_faces)
        # load mesh
        self.mesh = trimesh.load(mesh_file_name)
        # self.mesh.apply_scale(5.0)
        # print('bounding box:', self.mesh.bounding_box.bounds)
        vertices = np.array(self.mesh.vertices) - self.mesh.center_mass
        faces = np.array(self.mesh.faces)
        self.mass_center = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.mass_center[None] = self.mesh.center_mass
        print('mass_center:{}'.format(self.mass_center))
        body_sdf = np.load(options['sdf_file'])
        sdf_value = body_sdf['sdf_values']
        sdf_grad = np.array(np.gradient(sdf_value))
        sdf_grad = np.stack(sdf_grad, axis=-1)
        self.resolution = body_sdf['resolution']
        dx = np.array([body_sdf['dx'], body_sdf['dy'], body_sdf['dz']])
        self.dx = ti.Vector([dx[0], dx[1], dx[2]], dt=ti.f32)

        bounding_box = np.array(body_sdf['bbox'])
        self.bbox = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.bbox.from_numpy(bounding_box)
        self.sdf = ti.field(dtype=ti.f32, shape=sdf_value.shape)
        self.sdf.from_numpy(sdf_value)
        self.sdf_grad = ti.Vector.field(3, dtype=ti.f32, shape=(self.resolution, self.resolution, self.resolution))
        self.sdf_grad.from_numpy(sdf_grad)

        self.test_point_np = np.random.random() * (bounding_box[1] - bounding_box[0]) + bounding_box[0]
        self.test_point = ti.Vector([self.test_point_np[0], self.test_point_np[1], self.test_point_np[2]], dt=ti.f32)
        print('test point:', self.test_point)

        # static collision mesh
        self.static_mesh = ti.Vector.field(3, dtype=ti.f32, shape=vertices.shape[0])
        self.static_mesh.from_numpy(vertices)
        self.static_mesh_faces = ti.field(dtype=ti.i32, shape=faces.shape[0] * 3)
        self.static_mesh_faces.from_numpy(faces.flatten())

        if options['frames'] is not None:
            self.frames = options['frames']
        
        if options['transform'] is not None:
            self.initial_translation = ti.Vector(options['transform'][0:3], dt=ti.f32)
            self.initial_quat = ti.Vector(form_euler(options['transform'][3:6]), dt=ti.f32)
        else:
            self.initial_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
            self.initial_translation = ti.Vector([0.0, 0.0, 0.0])

        self.c, self.s = np.cos(np.deg2rad(options["slope_degree"])), np.sin(np.deg2rad(options["slope_degree"]))
        self.init_height = options['init_height']
        self.contact_normal = ti.Vector([-self.s, self.c, 0.0], dt=ti.f32)

        # conbvert mesh to taichi data structure
        self.x = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.x_t = ti.Vector.field(3, dtype=ti.f32)
        self.x_t2 = ti.Vector.field(3, dtype=ti.f32)
        ti.root.dense(ti.i, self.mesh.vertices.shape[0]).place(self.x, self.x.grad, self.x_t, self.x_t2)

        # translation 
        self.mass = ti.field(dtype=ti.f32, shape=())
        self.mass[None] = 0.0
        self.inv_mass = 0.0
        self.force = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)

        self.v = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.v2 = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)

        # rotation
        self.omega = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.omega2 = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)

        self.torque = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.torque2 = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)

        self.inertia = ti.Matrix.field(3, 3, dtype=ti.f32, needs_grad=True)

        # the indices is constant for all frames, so we don't need to store it in the frame loop, but only in the init_state function
        # and we won't need the grad of indices, so we don't need to set needs_grad=True
        self.indices = ti.field(dtype=ti.i32)

         # transformation information for rigid body
        self.quad = ti.Vector.field(4, dtype=ti.f32, needs_grad=True) 
        self.quad2 = ti.Vector.field(4, dtype=ti.f32, needs_grad=True)

        self.translation = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.translation2 = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)


        # place data
        ti.root.dense(ti.j, 3 * self.mesh.faces.shape[0]).place(self.indices)
        particles = ti.root.dense(ti.i, self.frames)
        # we only store these variables in the the mass center
        particles.place(self.torque, self.torque.grad, self.torque2, self.torque2.grad,
                         self.inertia, self.inertia.grad, 
                         self.quad, self.quad.grad, self.quad2, self.quad2.grad,
                         self.translation, self.translation.grad,
                         self.translation2, self.translation2.grad,
                        self.omega, self.omega.grad, self.omega2, self.omega2.grad,
                        self.v, self.v.grad, self.v2, self.v2.grad,
                        self.force, self.force.grad)

        # these variables are stored in each vertex
        # particles.dense(ti.j, self.mesh.vertices.shape[0]).place(self.x, self.x.grad)
        self.test_coord = ti.Vector.field(3, dtype=ti.f32, shape=10)

        self.init_state(vertices, faces)
        self.inv_mass = 1.0 / self.mass[None]

        # rigid body parameters
        self.gravity = ti.Vector([0.0, -3.0, 0.0])
        # params need to be optimized
        self.ke = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.mu = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.ke[None] = options['ke']
        self.mu[None] = options['mu']

        # set up ggui
        #create a window
        self.window = ti.ui.Window(name='Rigid body dynamics', res=(1280, 720), fps_limit=60, pos=(150,150))
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(1,2,3)
        self.camera.lookat(0,0,0)
        
        position = ti.Vector([1, 2, 3])
        lookat = ti.Vector([0, 0, 0])
        up = ti.Vector([0, 1, 0])
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        self.scene.set_camera(self.camera)

        self.pause = True

    @ti.kernel
    def calc_sdf_grad(self):
        for i,j,k in self.sdf:
            partial_x, partial_y, partial_z = 0., 0., 0.
            if i == 0:
                partial_x = (self.sdf[i + 1, j, k] - self.sdf[i,j,k]) / self.dx[None][0]
            elif i == self.resolution - 1:
                partial_x = (self.sdf[i, j, k] - self.sdf[i - 1,j,k]) / self.dx[None][0]
            if j == 0:
                partial_y = (self.sdf[i, j + 1, k] - self.sdf[i,j,k]) / self.dx[None][1]
            elif j == self.resolution - 1:
                partial_y = (self.sdf[i, j, k] - self.sdf[i,j - 1,k]) / self.dx[None][1]
            if k == 0:
                partial_z = (self.sdf[i, j, k + 1] - self.sdf[i,j,k]) / self.dx[None][2]
            elif k == self.resolution - 1:
                partial_z = (self.sdf[i, j, k] - self.sdf[i,j,k - 1]) / self.dx[None][2]
            if i != 0 and i != self.resolution - 1 and j != 0 and j != self.resolution - 1 and k != 0 and k != self.resolution - 1:
                partial_x = (self.sdf[i + 1, j, k] - self.sdf[i - 1,j,k]) / (2.0 * self.dx[None][0])
                partial_y = (self.sdf[i, j + 1, k] - self.sdf[i,j - 1,k]) / (2.0 * self.dx[None][1])
                partial_z = (self.sdf[i, j, k + 1] - self.sdf[i,j,k - 1]) / (2.0 * self.dx[None][2])

            self.sdf_grad[i, j, k] = ti.Vector([partial_x, partial_y, partial_z], dt=ti.f32).normalized()

    @ti.kernel
    def get_mesh_now(self):
        mat_R = quat_to_matrix(self.quad[0])
        mat_R2 = quat_to_matrix(self.quad2[0])
        for i in range(self.x.shape[0]):
            self.x_t[i] = self.translation[0] + mat_R @ self.x[i] + self.mass_center[None]
            self.x_t2[i] = self.translation2[0] + mat_R2 @ self.x[i] + self.mass_center[None]
            

    @ti.kernel
    def clear(self):
        self.force.fill(0.0)
        self.torque.fill(0.0)

    def run(self):
        data= {}
        data["frames"] = 60
        data["1_1_M"] = [[1.0, 0.0, 0.0,0.0], \
                        [0.0, 1.0, 0.0, 0.0], \
                        [0.0, 0.0, 1.0, -3.0], \
                        [0.0, 0.0, 0.0, 1.0]]
        data['dt'] = self.dt
        data['substep'] = self.substep
        data["results"] = []
        while self.window.running:
            if self.window.is_pressed(ti.ui.LEFT, 'b'):
                self.pause = not self.pause
            if not self.pause:
                for i in range(self.substep):
                    self.clear()
                    self.step(self.current_frame)
                new_frame = {}
                new_frame["frame_id"] = self.current_frame
                new_frame["obj1_translation"] = self.translation[0].to_numpy().tolist()
                new_frame["obj1_rotation"] = self.quad[0].to_numpy().tolist()
                new_frame['obj2_translation'] = self.translation2[0].to_numpy().tolist()
                new_frame['obj2_rotation'] = self.quad2[0].to_numpy().tolist()
                data["results"].append(new_frame)
                self.current_frame += 1

            #     # output mesh result
            #     self.get_mesh_now()
            #     mesh_vertices = self.x_t.to_numpy()
            #     mesh_vertices = mesh_vertices.tolist()
            #     mesh_faces = self.mesh.faces.tolist()
            #     mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
            #     mesh.export('mesh_result/obj1/{:04d}.obj'.format(self.current_frame))

            #     mesh_vertices = self.x_t2.to_numpy()
            #     mesh_vertices = mesh_vertices.tolist()
            #     mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
            #     mesh.export('mesh_result/obj2/{:04d}.obj'.format(self.current_frame))
            self.get_transform_matrix()
            self.render()
        # with open('transform1.json', 'w') as f:
        #     json.dump(data, f, indent=4)

    @ti.kernel
    def get_transform_matrix(self):
        R = quat_to_matrix(self.quad[0])
        T = ti.Matrix.identity(ti.f32, 4)
        T[0, 3] = self.translation[0][0]
        T[1, 3] = self.translation[0][1]
        T[2, 3] = self.translation[0][2]
        T[0:3, 0:3] = R
        self.T[0] = T

        R = quat_to_matrix(self.quad2[0])
        T = ti.Matrix.identity(ti.f32, 4)
        T[0, 3] = self.translation2[0][0]
        T[1, 3] = self.translation2[0][1]
        T[2, 3] = self.translation2[0][2]
        T[0:3, 0:3] = R
        self.T2[0] = T

    def render(self, frame=0):
        self.camera.track_user_inputs(self.window, movement_speed=0.05, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.8, 0.8, 0.8))
        self.scene.point_light(pos=(1,2,3), color=(1, 1, 1))
        # draw the floor
        self.scene.mesh(self.floor_vertices, self.floor_faces, color=(0.5, 0.5, 0.5),show_wireframe=True)
        self.scene.mesh_instance(self.x, self.indices, color=(0.5, 0.5, 0.5),show_wireframe=True, transforms=self.T)
        self.scene.mesh_instance(self.x, self.indices, color=(0.5, 0.5, 0.5),show_wireframe=False, transforms=self.T2)
        # draw static mesh
        # self.scene.mesh(self.static_mesh, self.static_mesh_faces, color=(0.3, 0.2, 0.1),show_wireframe=False)
        # self.scene.mesh(self.x, self.indices, color=(0.5, 0.5, 0.5),show_wireframe=True)
        self.canvas.scene(self.scene)
        self.window.show()
        # print('mass center:', self.mass_center[None] + self.translation[0])

    
    @ti.kernel
    def init_state(self, vertices:ti.types.ndarray(), faces:ti.types.ndarray()):
        for i in range(vertices.shape[0]):
            self.x[i] = ti.Vector([vertices[i, 0], vertices[i, 1], vertices[i, 2]], dt=ti.f32)
        for i in range(faces.shape[0]):
            for j in ti.static(range(3)):
                self.indices[i * 3 + j] = faces[i, j]
        # set initial transformation
        self.quad[0] = self.initial_quat
        self.translation[0] = ti.Vector([0.0, 0.0, 0.0]) + self.initial_translation

        self.quad2[0] = self.initial_quat
        self.translation2[0] = ti.Vector([0.0, -0.2, 0.0]) + self.initial_translation

        self.v[0] = ti.Vector([0.0, 0.0, 0.0])
        self.v2[0] = ti.Vector([0.0, 0.0, 0.0])
        # set initial rotation
        self.omega[0] = ti.Vector([0.0, 0.0, 0.0])
        self.omega2[0] = ti.Vector([0.0, 0.0, 0.0])

        self.inertia[0] = ti.Matrix.zero(ti.f32, 3, 3)

        # calculate ref inertia (frame 0)
        mass = 1.0
        for i in range(vertices.shape):
            ti.atomic_add(self.mass[None], mass)
            r = self.x[i] - self.mass_center[None]
            # inertia = \sum_{i=1}^{n} m_i (r_i^T r_i I - r_i r_i^T)  https://en.wikipedia.org/wiki/List_of_moments_of_inertia
            # as r_i is a col vector, r_i^T is a row vector, so r_i^T r_i is a scalar (actually is dot product)
            I_i = mass * (r.dot(r) * ti.Matrix.identity(ti.f32, 3) - r.outer_product(r))
            ti.atomic_add(self.inertia[0], I_i)
        print('inertia:', self.inertia[0])

    @ti.func 
    def get_sdf(self, x)->ti.f32:
        # translate the x to the sdf coordinate
        lower_bound = self.bbox[0]
        loca_x = x - lower_bound
        # find the nearest grid
        grid_x = ti.Vector([0, 0, 0], dt=ti.i32)
        grid_x[0] = ti.cast(loca_x[0] / self.dx[0], ti.i32)
        grid_x[1] = ti.cast(loca_x[1] / self.dx[1], ti.i32)
        grid_x[2] = ti.cast(loca_x[2] / self.dx[2], ti.i32)
        # tri-linear interpolation
        # find the 8 nearest grid
        # Adjust grid indices at the boundaries
        grid_offset = ti.Vector([1, 1, 1])
        for i in ti.static(range(3)):
            if grid_x[i] == self.sdf.shape[i] - 1:
                grid_offset[i] = -1
        grid_000 = grid_x
        grid_001 = grid_x + ti.Vector([0, 0, grid_offset[2]])
        grid_010 = grid_x + ti.Vector([0, grid_offset[1], 0])
        grid_011 = grid_x + ti.Vector([0, grid_offset[1], grid_offset[2]])
        grid_100 = grid_x + ti.Vector([grid_offset[0], 0, 0])
        grid_101 = grid_x + ti.Vector([grid_offset[0], 0, grid_offset[2]])
        grid_110 = grid_x + ti.Vector([grid_offset[0], grid_offset[1], 0])
        grid_111 = grid_x + grid_offset
        # calculate the weight
        weight = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        for i in ti.static(range(3)):
            weight[i] = (loca_x[i] / self.dx[i]) - grid_x[i]
        # calculate the sdf
        sdf_000 = self.sdf[grid_000[0], grid_000[1], grid_000[2]]
        sdf_001 = self.sdf[grid_001[0], grid_001[1], grid_001[2]]
        sdf_010 = self.sdf[grid_010[0], grid_010[1], grid_010[2]]
        sdf_011 = self.sdf[grid_011[0], grid_011[1], grid_011[2]]
        sdf_100 = self.sdf[grid_100[0], grid_100[1], grid_100[2]]
        sdf_101 = self.sdf[grid_101[0], grid_101[1], grid_101[2]]
        sdf_110 = self.sdf[grid_110[0], grid_110[1], grid_110[2]]
        sdf_111 = self.sdf[grid_111[0], grid_111[1], grid_111[2]]
        sdf_00 = sdf_000 * (1.0 - weight[0]) + sdf_100 * weight[0]
        sdf_01 = sdf_001 * (1.0 - weight[0]) + sdf_101 * weight[0]
        sdf_10 = sdf_010 * (1.0 - weight[0]) + sdf_110 * weight[0]
        sdf_11 = sdf_011 * (1.0 - weight[0]) + sdf_111 * weight[0]
        sdf_0 = sdf_00 * (1.0 - weight[1]) + sdf_10 * weight[1]
        sdf_1 = sdf_01 * (1.0 - weight[1]) + sdf_11 * weight[1]
        sdf = sdf_0 * (1.0 - weight[2]) + sdf_1 * weight[2]
        return sdf


    @ti.kernel
    def step(self, f:ti.f32):
        self.v[0] += self.dt * self.gravity
        self.v[0] *= 0.999
        self.omega[0] *= 0.998

        self.v2[0] += self.dt * self.gravity
        self.v2[0] *= 0.999
        self.omega2[0] *= 0.998

        # two obj collision
        mat_R1 = quat_to_matrix(self.quad[0])
        mat_R2 = quat_to_matrix(self.quad2[0])

        num_collision = 0
        avg_collision_point = ti.Vector([0.0, 0.0, 0.0])
        for i in range(self.x.shape[0]):
            # transform obj2 to the world coordinate system
            ri = self.x[i]
            xi = self.translation2[0] + mat_R2 @ ri + self.mass_center[None]
            # transform xi to the obj1 coordinate system
            xi = xi - self.translation[0]
            xi = mat_R1.transpose() @ xi
            if xi[0] > self.bbox[0][0] and xi[0] < self.bbox[1][0] and xi[1] > self.bbox[0][1] and \
            xi[1] < self.bbox[1][1] and xi[2] > self.bbox[0][2] and xi[2] < self.bbox[1][2]:
                sdf = self.get_sdf(xi)
                if sdf < 0.0:
                    ti.atomic_add(num_collision, 1)
                    ti.atomic_add(avg_collision_point, ri)
        if num_collision > 0:
            ri = avg_collision_point / num_collision
            Rri1 = mat_R1 @ ri
            Rri2 = mat_R2 @ ri

            vi1 = self.v[0] + self.omega[0].cross(Rri1)
            vi2 = self.v2[0] + self.omega2[0].cross(Rri2)
            
            if ti.math.dot(vi1, vi2) < 0:
                vi1_new = vi2
                vi2_new = vi1

                I1 = mat_R1 @ self.inertia[0] @ mat_R1.transpose()
                I2 = mat_R2 @ self.inertia[0] @ mat_R2.transpose()

                Rri1_mat = Get_Cross_Matrix(Rri1)
                Rri2_mat = Get_Cross_Matrix(Rri2)

                k1 = ti.Matrix([[self.inv_mass, 0.0, 0.0],
                                [0.0, self.inv_mass, 0.0],
                                [0.0, 0.0, self.inv_mass]]) - Rri1_mat @ I1.inverse() @ Rri1_mat
                k2 = ti.Matrix([[self.inv_mass, 0.0, 0.0],
                                [0.0, self.inv_mass, 0.0],
                                [0.0, 0.0, self.inv_mass]]) - Rri2_mat @ I2.inverse() @ Rri2_mat
                
                J1 = k1.inverse() @ (vi1_new - vi1)
                J2 = k2.inverse() @ (vi2_new - vi2)

                self.v[0] += self.inv_mass * J1
                self.omega[0] += I1.inverse() @ Rri1_mat @ J1

                self.v2[0] += self.inv_mass * J2
                self.omega2[0] += I2.inverse() @ Rri2_mat @ J2

        # # collision Impulse
        # avg_collision_point = ti.Vector([0.0, 0.0, 0.0])
        # num_collision = 0
        mat_R = quat_to_matrix(self.quad[0])
        # # implus model
        # for i in range(self.x.shape[0]):
        #     ri = self.x[i]
        #     xi = self.translation[0] + mat_R @ ri + self.mass_center[None]
        #     if xi.dot(self.contact_normal) < -self.c * self.init_height:
        #         vi = self.v[0] + self.omega[0].cross(mat_R @ ri)
        #         if vi.dot(self.contact_normal) < 0.0:
        #             ti.atomic_add(num_collision, 1)
        #             ti.atomic_add(avg_collision_point, ri)
        # if num_collision > 0:
        #     ri = avg_collision_point / num_collision
        #     Rri = mat_R @ ri
        #     vi = self.v[0] + self.omega[0].cross(mat_R @ ri)

        #     # calculate new velocity
        #     v_iN = vi.dot(self.contact_normal) * self.contact_normal
        #     v_iT = vi - v_iN
        #     # impluse method 
        #     alpha = 1.0 - (self.mu[None] * (1.0 + self.ke[None]) * (v_iN.norm()/v_iT.norm()))
        #     if alpha < 0.0:
        #         alpha = 0.0
        #     vi_new_N = -self.ke[None] * v_iN
        #     vi_new_T = alpha * v_iT
        #     print('f: {}, apha:{}, vt_new:{}, vt:{}, item: {}'.format(f, alpha, vi_new_T, v_iT, self.mu[None] * (1.0 + self.ke[None]) * (v_iN.norm()/v_iT.norm())))
        #     vi_new = vi_new_N + vi_new_T
        #     # calculate impulse
        #     I = mat_R @ self.inertia[0] @ mat_R.transpose()
        #     Rri_mat = Get_Cross_Matrix(Rri)
        #     k = ti.Matrix([[self.inv_mass, 0.0, 0.0],\
        #                    [0.0, self.inv_mass, 0.0],\
        #                    [0.0, 0.0, self.inv_mass]]) - Rri_mat @ I.inverse() @ Rri_mat
        #     J = k.inverse() @ (vi_new - vi)
        #     # update velocity
        #     self.v[0]+= self.inv_mass * J
        #     self.omega[0] += I.inverse() @ Rri_mat @ J

        planar_contat_normal = ti.Vector([0.0, 1.0, 0.0])
        num_collision = 0
        avg_collision_point = ti.Vector([0.0, 0.0, 0.0])
        for i in range(self.x.shape[0]):
            ri = self.x[i]
            xi = self.translation[0] + mat_R @ ri + self.mass_center[None]
            if xi.dot(planar_contat_normal) < 0.01:
                vi = self.v[0] + self.omega[0].cross(mat_R @ ri)
                if vi.dot(planar_contat_normal) < 0.0:
                    ti.atomic_add(num_collision, 1)
                    ti.atomic_add(avg_collision_point, ri)
        if num_collision > 0:
            ri = avg_collision_point / num_collision
            Rri = mat_R @ ri
            vi = self.v[0] + self.omega[0].cross(Rri)
            # calculate new velocity
            v_iN = vi.dot(planar_contat_normal) * planar_contat_normal
            v_iT = vi - v_iN
            # impluse method 
            alpha = 1.0 - (self.mu[None] * (1.0 + self.ke[None]) * (v_iN.norm()/v_iT.norm()))
            if alpha < 0.0:
                alpha = 0.0
            vi_new_N = -self.ke[None] * v_iN
            vi_new_T = alpha * v_iT
            vi_new = vi_new_N + vi_new_T

            # calculate impulse
            I = mat_R @ self.inertia[0] @ mat_R.transpose()
            Rri_mat = Get_Cross_Matrix(Rri)
            k = ti.Matrix([[self.inv_mass, 0.0, 0.0],\
                           [0.0, self.inv_mass, 0.0],\
                           [0.0, 0.0, self.inv_mass]]) - Rri_mat @ I.inverse() @ Rri_mat
            J = k.inverse() @ (vi_new - vi)
            # update velocity
            self.v[0]+= self.inv_mass * J
            self.omega[0] += I.inverse() @ Rri_mat @ J
        

        #obj2 planar collision
        num_collision = 0
        avg_collision_point = ti.Vector([0.0, 0.0, 0.0])
        mat_R = quat_to_matrix(self.quad2[0])
        for i in range(self.x.shape[0]):
            ri = self.x[i]
            xi = self.translation2[0] + mat_R @ ri + self.mass_center[None]
            if xi.dot(planar_contat_normal) < 0.01:
                vi = self.v2[0] + self.omega2[0].cross(mat_R @ ri)
                if vi.dot(planar_contat_normal) < 0.0:
                    ti.atomic_add(num_collision, 1)
                    ti.atomic_add(avg_collision_point, ri)
        if num_collision > 0:
            ri = avg_collision_point / num_collision
            Rri = mat_R @ ri
            vi = self.v2[0] + self.omega2[0].cross(Rri)
            # calculate new velocity
            v_iN = vi.dot(planar_contat_normal) * planar_contat_normal
            v_iT = vi - v_iN
            # impluse method 
            alpha = 1.0 - (self.mu[None] * (1.0 + self.ke[None]) * (v_iN.norm()/v_iT.norm()))
            if alpha < 0.0:
                alpha = 0.0
            vi_new_N = -self.ke[None] * v_iN
            vi_new_T = alpha * v_iT
            vi_new = vi_new_N + vi_new_T

            # calculate impulse
            I = mat_R @ self.inertia[0] @ mat_R.transpose()
            Rri_mat = Get_Cross_Matrix(Rri)
            k = ti.Matrix([[self.inv_mass, 0.0, 0.0],\
                           [0.0, self.inv_mass, 0.0],\
                           [0.0, 0.0, self.inv_mass]]) - Rri_mat @ I.inverse() @ Rri_mat
            J = k.inverse() @ (vi_new - vi)
            # update velocity
            self.v2[0]+= self.inv_mass * J
            self.omega2[0] += I.inverse() @ Rri_mat @ J

        wt = self.omega[0] * self.dt * 0.5
        wt2 = self.omega2[0] * self.dt * 0.5

        dq = quat_mul(ti.Vector([0.0, wt[0], wt[1], wt[2]]), self.quad[0])
        dq2 = quat_mul(ti.Vector([0.0, wt2[0], wt2[1], wt2[2]]), self.quad2[0])

        self.translation[0] += self.dt * self.v[0]
        self.translation2[0] += self.dt * self.v2[0]

        self.quad[0] += dq
        self.quad2[0] += dq2

        self.quad[0] = self.quad[0].normalized()
        self.quad2[0] = self.quad2[0].normalized()

    @ti.kernel
    def step_soft_contact(self, f:ti.f32):
        self.v[0] += self.dt * self.gravity
        self.v[0] *= 0.999
        self.omega[0] *= 0.998

        # collision forces
        force = ti.Vector([0.0, 0.0, 0.0])
        torque = ti.Vector([0.0, 0.0, 0.0])
        mat_R = quat_to_matrix(self.quad[0])

        # soft contact parameters
        kn = 1e4 # Spring constant
        kf = 0.25    # Damping coefficient
        mu = 1e4
        for i in range(self.x.shape[0]):
            ri = self.x[i] - self.mass_center[None]
            xi = self.translation[0] + mat_R @ ri + self.mass_center[None]
            Rri = mat_R @ ri
            # Check for collision with the plane\
            planar_contat_normal = ti.Vector([0.0, 1.0, 0.0])
            d = planar_contat_normal.dot(xi) - 0.01
            if d < 0.0:
                fn_mag = kn * ti.math.pow(-ti.min(d, 0.0), 3 - 1)
                fn = fn_mag * planar_contat_normal
                # compute the friction force
                vi = self.v[0] + self.omega[0].cross(mat_R @ ri)
                us = vi - vi.dot(planar_contat_normal) * planar_contat_normal
                us_mag = us.norm()
                ff = -ti.min(kf, mu* fn_mag / (us_mag + 1e-6)) * us
                force += (fn + ff)
                torque += Rri.cross(fn + ff)
        I = mat_R @ self.inertia[0] @ mat_R.transpose()
        # Update linear and angular velocity
        self.v[0] += self.dt * self.inv_mass * force
        self.omega[0] += self.dt * I.inverse() @ torque

        # Integrate position and orientation
        self.translation[0] += self.dt * self.v[0]
        wt = self.omega[0] * self.dt * 0.5
        dq = quat_mul(ti.Vector([0.0, wt[0], wt[1], wt[2]]), self.quad[0])

        self.quad[0] += dq
        self.quad[0] = self.quad[0].normalized()


    @ti.kernel
    def test_sdf_tri(self):
        # get  a random point in bbox
        position = self.test_point
        if position[0] > self.bbox[0][0] and position[0] < self.bbox[1][0] and position[1] > self.bbox[0][1] and \
            position[1] < self.bbox[1][1] and position[2] > self.bbox[0][2] and position[2] < self.bbox[1][2]:
            sdf_value = self.get_sdf(position)
            print('sdf value', sdf_value)

    def real_sdf_value(self):
        test_point = self.test_point_np
        test_point = test_point.reshape(1, 3)
        import mesh_to_sdf
        sdf_value = mesh_to_sdf.mesh_to_sdf(self.mesh, test_point, surface_point_method='scan', sign_method='normal', \
        bounding_radius=None, scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
        print('real sdf_value:', sdf_value)

if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_name = Path(current_directory) / 'assets' / 'bunny_original.obj'
    sdf_file_name = Path(current_directory)/'bunny_origin.npz'

    robot = rigid_body(file_name, { 'frames': 100 ,
                                    'ke': 0.2,
                                    'kn': 0.2,
                                    'mu': 0.5,
                                    'init_height': 1.0,
                                    'offset': 0.0, 
                                    'p' : 3,
                                    'slope_degree': 30.0,
                                    'transform': [2.1, 0.5, 0.25, -90.0, -90.0, 0.0],
                                    'sdf_file' : sdf_file_name
                                    })
    robot.run()
    # robot.real_sdf_value()
    # robot.test_sdf_tri()
    