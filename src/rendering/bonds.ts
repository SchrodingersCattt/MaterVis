/**
 * Bond rendering utilities
 */

import * as THREE from 'three'
import { Bond } from '../utils/bonds'
import { Site } from '../crystal'

/**
 * Create bond mesh with plastic material
 * @param site1 First site
 * @param site2 Second site
 * @param radius Bond radius
 * @returns THREE.Mesh instance
 */
export function createBondMesh(site1: Site, site2: Site, radius: number = 0.1): THREE.Mesh {
  if (!site1.cartesian || !site2.cartesian) {
    throw new Error('Sites must have cartesian coordinates')
  }
  
  // Calculate direction vector and distance
  const start = new THREE.Vector3(...site1.cartesian)
  const end = new THREE.Vector3(...site2.cartesian)
  const direction = new THREE.Vector3().subVectors(end, start)
  const distance = direction.length()
  
  // Create cylinder geometry
  const geometry = new THREE.CylinderGeometry(radius, radius, distance, 8, 1)
  geometry.translate(0, distance / 2, 0)
  geometry.rotateX(Math.PI / 2)
  
  // Create plastic material
  const material = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(0x808080), // Medium gray
    metalness: 0.0,
    roughness: 0.3, // Slightly higher roughness than atoms
    clearcoat: 1.0,
    clearcoatRoughness: 0.15,
    envMapIntensity: 0.4
  })
  
  const bondMesh = new THREE.Mesh(geometry, material)
  
  // Position and orient the bond
  bondMesh.position.copy(start)
  bondMesh.lookAt(end)
  
  return bondMesh
}

/**
 * Create instanced bonds for better performance
 * @param bonds Array of bonds
 * @param sites Array of atomic sites
 * @param radius Bond radius
 * @returns THREE.InstancedMesh instance
 */
export function createInstancedBonds(bonds: Bond[], sites: Site[], radius: number = 0.1): THREE.InstancedMesh {
  if (bonds.length === 0) return new THREE.InstancedMesh(new THREE.BufferGeometry(), new THREE.Material(), 0)
  
  // Calculate maximum bond length to set up geometry
  let maxLength = 0
  bonds.forEach(bond => {
    const site1 = sites[bond.atom1]
    const site2 = sites[bond.atom2]
    if (site1.cartesian && site2.cartesian) {
      const start = new THREE.Vector3(...site1.cartesian)
      const end = new THREE.Vector3(...site2.cartesian)
      const distance = start.distanceTo(end)
      maxLength = Math.max(maxLength, distance)
    }
  })
  
  // Create cylinder geometry
  const geometry = new THREE.CylinderGeometry(radius, radius, maxLength, 8, 1)
  geometry.translate(0, maxLength / 2, 0)
  geometry.rotateX(Math.PI / 2)
  
  // Create plastic material
  const material = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(0x808080), // Medium gray
    metalness: 0.0,
    roughness: 0.3, // Slightly higher roughness than atoms
    clearcoat: 1.0,
    clearcoatRoughness: 0.15,
    envMapIntensity: 0.4
  })
  
  const instancedMesh = new THREE.InstancedMesh(geometry, material, bonds.length)
  
  // Set transformation matrix for each bond
  const matrix = new THREE.Matrix4()
  const quaternion = new THREE.Quaternion()
  const scale = new THREE.Vector3(1, 1, 1)
  
  bonds.forEach((bond, index) => {
    const site1 = sites[bond.atom1]
    const site2 = sites[bond.atom2]
    
    if (site1.cartesian && site2.cartesian) {
      const start = new THREE.Vector3(...site1.cartesian)
      const end = new THREE.Vector3(...site2.cartesian)
      const direction = new THREE.Vector3().subVectors(end, start)
      const distance = direction.length()
      
      // Position at midpoint
      const position = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5)
      
      // Calculate orientation
      const yAxis = new THREE.Vector3(0, 1, 0)
      quaternion.setFromUnitVectors(yAxis, direction.clone().normalize())
      
      // Scale by distance
      scale.y = distance / maxLength
      
      matrix.compose(position, quaternion, scale)
      instancedMesh.setMatrixAt(index, matrix)
    }
  })
  
  return instancedMesh
}