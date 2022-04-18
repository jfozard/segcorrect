

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "Python.h"

#include <vector>
#include <math.h>

#define LARGE 1e20


typedef Eigen::Vector3f Vec3;

float ray_tri(const Vec3& o, const Vec3& d,
	      const Vec3& v0, const Vec3& v1, const Vec3& v2)
{

  const float eps = 1e-2;
  const float eps1 = 1e-12;
  
  Vec3 e1 = v1 - v0;
  Vec3 e2 = v2 - v0;
  Vec3 p = d.cross(e2);

  float det = e1.dot(p);
  if((-eps1 < det) && (det < eps1))
    return LARGE;
  //  float inv_det = 1.0/det;
  
  Vec3 t = o - v0;
  float u = t.dot(p)/det;
  if(u<0.0 || u>1.0)
    return LARGE;

  Vec3 q = t.cross(e1);
  float v = d.dot(q)/det;
  if(v<0.0 || (u+v)>1.0)
    return LARGE;

  float time = e2.dot(q)/det;
  return time;
}

extern "C"
int intersect(float verts[][3], int tris[][3], int Nv, int Nt, float start[], float end[])
{
  int closest_tri = -1;
  float closest_time = LARGE;

  Vec3 o(start);
  Vec3 e(end);

  Vec3 d = e - o;

  for(int i=0; i<Nt; i++)
    {
      float t = ray_tri(o, d, 
			Eigen::Map<Vec3>(verts[tris[i][0]]),
			Eigen::Map<Vec3>(verts[tris[i][1]]),
			Eigen::Map<Vec3>(verts[tris[i][2]]));
      if(t < closest_time && ((0.0<t) && (t<1.0))) 
	{
	  closest_time = t;
	  closest_tri = i;
	}
    }

  return closest_tri;
}

extern "C"
float intersect_dist(float verts[][3], int tris[][3], int Nv, int Nt, float start[], float end[])
{
  float closest_time = LARGE;

  Vec3 o(start);
  Vec3 e(end);

  Vec3 d = e - o;

  for(int i=0; i<Nt; i++)
    {
      float t = ray_tri(o, d, 
			Eigen::Map<Vec3>(verts[tris[i][0]]),
			Eigen::Map<Vec3>(verts[tris[i][1]]),
			Eigen::Map<Vec3>(verts[tris[i][2]]));
      if(t < closest_time && ((0.0<t) && (t<1.0))) 
	{
	  closest_time = t;
	}
    }
  if(closest_time < LARGE)
    return closest_time;
  else
    return NAN;
}

extern "C"
float _stack_from_mesh(float verts[][3], unsigned int tris[][3], int Nv, int Nt, float corner[], float spacing[], char stack[], Py_ssize_t shape[], Py_ssize_t strides[])
{
  const float eps = 1e-6;
  Vec3 p_corner(corner);
  Vec3 p_spacing(spacing);
  Vec3 p_dir(0.0, 0.0, spacing[2]);
  for(int i1=0; i1<shape[1]; i1++)
    #pragma omp parallel for
    for(int i2=0; i2<shape[2]; i2++)
      {
	for (int i0=0; i0<shape[0]; i0++)
	  {
	    stack[i0*strides[0]+i1*strides[1]+i2*strides[2]] = 0;
	  }
	// Find start of ray to shoot
	Vec3 p_base = p_corner;
	p_base[0] += i2*spacing[0];
	p_base[1] += i1*spacing[1];
	//
	std::vector<float> t_list;

	for(int i=0; i<Nt; i++)
	    {
	      float t = ray_tri(p_base, p_dir, 
				Eigen::Map<Vec3>(verts[tris[i][0]]),
				Eigen::Map<Vec3>(verts[tris[i][1]]),
				Eigen::Map<Vec3>(verts[tris[i][2]]));
	      if((t<LARGE) && (t>0))
		{
		  bool in_list = false;
		  for(auto t2 : t_list)
		    {
		      if(std::abs(t - t2) < eps)
			{
			  in_list = true;
			  break;
			}		    
		    }
		  if(!in_list)
		    {
		      t_list.push_back(t);
		      
		      for (int i0=0; ((i0<t) && (i0<shape[2])); i0++)
			{
			  stack[i0*strides[0]+i1*strides[1]+i2*strides[2]] = !stack[i0*strides[0]+i1*strides[1]+i2*strides[2]];
			}
		    }
		}
	    }
      }
}
