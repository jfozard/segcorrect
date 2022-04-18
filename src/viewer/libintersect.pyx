
cdef extern int intersect(float *verts, unsigned int *tris, int Nv, int Nt, float *start, float *end)

cpdef int intersect_mesh_ray(float[:,:] verts, unsigned int [:,:] tris, float[:] start, float[:] end):
        return intersect(&verts[0,0], &tris[0,0], verts.shape[0], tris.shape[0], &start[0], &end[0]) 


cdef extern float intersect_dist(float *verts, unsigned int *tris, int Nv, int Nt, float *start, float *end)

cpdef float intersect_mesh_ray_dist(float[:,:] verts, unsigned int [:,:] tris, float[:] start, float[:] end):
        return intersect_dist(&verts[0,0], &tris[0,0], verts.shape[0], tris.shape[0], &start[0], &end[0]) 


cdef extern float _stack_from_mesh(float *verts, unsigned int *tris, int Nv, int Nt, float *corner, float *spacing, char *stack, Py_ssize_t *shape, Py_ssize_t *strides)

cpdef stack_from_mesh(float[:,:] verts, unsigned int [:,:] tris, float[:] corner, float[:] spacing, char[:,:,:] stack):
        _stack_from_mesh(&verts[0,0], &tris[0,0], verts.shape[0], tris.shape[0], &corner[0], &spacing[0], &stack[0,0,0], &stack.shape[0], &stack.strides[0]) 

