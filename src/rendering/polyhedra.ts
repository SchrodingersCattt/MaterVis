/**
 * Polyhedra rendering utilities
 */

import * as THREE from 'three'
import { Site } from '../crystal'

/**
 * Create coordination polyhedron mesh
 * @param center Central atom site
 * @param neighbors Array of neighboring atom sites
 * @param faceAlpha Transparency for polyhedron faces (0-1)
 * @returns THREE.Group containing faces and edges
 */
export function createPolyhedronMesh(
  center: Site, 
  neighbors: Site[], 
  faceAlpha: number = 0.25
): THREE.Group {
  const group = new THREE.Group()
  
  if (!center.cartesian) {
    throw new Error('Center site must have cartesian coordinates')
  }
  
  // Check that all neighbors have cartesian coordinates
  if (!neighbors.every(n => n.cartesian)) {
    throw new Error('All neighbor sites must have cartesian coordinates')
  }
  
  // For a simple implementation, we'll create a convex hull if we have enough points
  if (neighbors.length < 4) {
    // Not enough neighbors to make a polyhedron
    return group
  }
  
  // Extract neighbor positions
  const positions: THREE.Vector3[] = neighbors
    .map(n => new THREE.Vector3(...n.cartesian!))
  
  // Create faces (simplified implementation)
  // In a full implementation, we would compute the actual convex hull
  try {
    // Create transparent faces
    const faceMaterial = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(0x40a0ff),
      metalness: 0.0,
      roughness: 0.25,
      clearcoat: 1.0,
      clearcoatRoughness: 0.1,
      envMapIntensity: 0.3,
      transparent: true,
      opacity: faceAlpha
    })
    
    // Create a simple approximation using a sphere for now
    const centerPos = new THREE.Vector3(...center.cartesian)
    let avgDistance = 0
    positions.forEach(pos => {
      avgDistance += centerPos.distanceTo(pos)
    })
    avgDistance /= positions.length
    
    const faceGeometry = new THREE.SphereGeometry(avgDistance * 0.8, 8, 8)
    const faces = new THREE.Mesh(faceGeometry, faceMaterial)
    faces.position.copy(centerPos)
    group.add(faces)
    
    // Create glossy edges
    const edgeMaterial = new THREE.LineBasicMaterial({
      color: new THREE.Color(0xffffff),
      linewidth: 2
    })
    
    // Simple edge representation
    const edgeGeometry = new THREE.SphereGeometry(avgDistance * 0.8, 8, 8)
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(edgeGeometry), 
      edgeMaterial
    )
    edges.position.copy(centerPos)
    group.add(edges)
  } catch (e) {
    console.warn('Could not create polyhedron mesh:', e)
  }
  
  return group
}

/**
 * Identify polyhedron type based on coordination number and geometry
 * @param neighbors Array of neighboring atom sites
 * @returns String describing polyhedron type
 */
export function identifyPolyhedronType(neighbors: Site[]): string {
  const coordinationNumber = neighbors.length
  
  // Simple identification based on coordination number
  switch (coordinationNumber) {
    case 2:
      return 'Linear'
    case 3:
      return 'Trigonal'
    case 4:
      return 'Tetrahedral'
    case 5:
      return 'Trigonal bipyramidal'
    case 6:
      return 'Octahedral'
    case 8:
      return 'Square antiprismatic'
    default:
      return `${coordinationNumber}-coordinate`
  }
}