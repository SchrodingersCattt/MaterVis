/**
 * Atom rendering utilities
 */

import * as THREE from 'three'
import { Site } from '../crystal'

// CPK color scheme (with muted tones for plastic aesthetic)
const ATOM_COLORS: { [element: string]: THREE.Color } = {
  'H': new THREE.Color(0xf2f2f2),   // White
  'He': new THREE.Color(0xd9ffff),  // Pale cyan
  'Li': new THREE.Color(0xcc80ff),  // Purple
  'Be': new THREE.Color(0xc2ff00),  // Lime
  'B': new THREE.Color(0xffb5b5),   // Pink
  'C': new THREE.Color(0x909090),   // Dark gray
  'N': new THREE.Color(0x3050f8),   // Blue
  'O': new THREE.Color(0xff0d0d),   // Red
  'F': new THREE.Color(0x90e050),   // Green
  'Ne': new THREE.Color(0xb3e3f5),  // Light blue
  'Na': new THREE.Color(0xab5cf2),  // Purple
  'Mg': new THREE.Color(0x8aff00),  // Lime
  'Al': new THREE.Color(0xbfa6a6),  // Light gray
  'Si': new THREE.Color(0xf0c8a0),  // Tan
  'P': new THREE.Color(0xff8000),   // Orange
  'S': new THREE.Color(0xffff30),   // Yellow
  'Cl': new THREE.Color(0x1ff01f),  // Green
  'Ar': new THREE.Color(0x80d1e3),  // Light blue
  'K': new THREE.Color(0x8f40d4),   // Purple
  'Ca': new THREE.Color(0x3dff00),  // Lime
  'Sc': new THREE.Color(0xe6e6e6),  // Light gray
  'Ti': new THREE.Color(0xbfc2c7),  // Gray
  'V': new THREE.Color(0xa6a6ab),   // Gray
  'Cr': new THREE.Color(0x8a99c7),  // Blue-gray
  'Mn': new THREE.Color(0x9c7ac7),  // Purple
  'Fe': new THREE.Color(0xe06633),  // Orange
  'Co': new THREE.Color(0xf090a0),  // Pink
  'Ni': new THREE.Color(0x50d050),  // Green
  'Cu': new THREE.Color(0xc88033),  // Orange
  'Zn': new THREE.Color(0x7d80b0),  // Blue-gray
  'Ga': new THREE.Color(0xc28f8f),  // Brown
  'Ge': new THREE.Color(0x668f8f),  // Gray
  'As': new THREE.Color(0xbd80e3),  // Purple
  'Se': new THREE.Color(0xffa100),  // Orange
  'Br': new THREE.Color(0xa62929),  // Red
  'Kr': new THREE.Color(0x5cb8d1),  // Blue
  // Default color for unknown elements
  'X': new THREE.Color(0xaaaaaa)    // Medium gray
}

// Atom radii (in Å)
const ATOM_RADII: { [element: string]: number } = {
  'H': 0.31,
  'He': 0.31,
  'Li': 1.28,
  'Be': 0.96,
  'B': 0.84,
  'C': 0.76,
  'N': 0.71,
  'O': 0.66,
  'F': 0.57,
  'Ne': 0.58,
  'Na': 1.66,
  'Mg': 1.41,
  'Al': 1.21,
  'Si': 1.11,
  'P': 1.07,
  'S': 1.05,
  'Cl': 1.02,
  'Ar': 1.06,
  'K': 2.03,
  'Ca': 1.76,
  'Sc': 1.70,
  'Ti': 1.60,
  'V': 1.53,
  'Cr': 1.39,
  'Mn': 1.39,
  'Fe': 1.32,
  'Co': 1.26,
  'Ni': 1.24,
  'Cu': 1.32,
  'Zn': 1.22,
  'Ga': 1.22,
  'Ge': 1.20,
  'As': 1.19,
  'Se': 1.20,
  'Br': 1.20,
  'Kr': 1.16,
  // Default radius for unknown elements
  'X': 1.0
}

/**
 * Get color for an element
 * @param element Element symbol
 * @returns THREE.Color instance
 */
export function getAtomColor(element: string): THREE.Color {
  return ATOM_COLORS[element] || ATOM_COLORS['X']
}

/**
 * Get radius for an element
 * @param element Element symbol
 * @returns Radius in Å
 */
export function getAtomRadius(element: string): number {
  return ATOM_RADII[element] || ATOM_RADII['X']
}

/**
 * Create atom mesh with plastic material
 * @param element Element symbol
 * @param radius Scale factor for atom radius
 * @returns THREE.Mesh instance
 */
export function createAtomMesh(element: string, radiusScale: number = 1.0): THREE.Mesh {
  const radius = getAtomRadius(element) * radiusScale
  const geometry = new THREE.IcosahedronGeometry(radius, 3)
  const material = new THREE.MeshPhysicalMaterial({
    color: getAtomColor(element),
    metalness: 0.0,
    roughness: 0.22,
    clearcoat: 1.0,
    clearcoatRoughness: 0.15,
    envMapIntensity: 0.5
  })
  
  return new THREE.Mesh(geometry, material)
}

/**
 * Create instanced atoms for better performance
 * @param sites Array of atomic sites
 * @param radiusScale Scale factor for atom radii
 * @returns THREE.InstancedMesh instance
 */
export function createInstancedAtoms(sites: Site[], radiusScale: number = 1.0): THREE.InstancedMesh {
  // For simplicity, we'll create one instanced mesh per element
  // In a full implementation, we would batch similar materials together
  
  // Find the most common element to set up the instanced mesh
  const elementCounts: { [element: string]: number } = {}
  sites.forEach(site => {
    elementCounts[site.element] = (elementCounts[site.element] || 0) + 1
  })
  
  const mostCommonElement = Object.keys(elementCounts).reduce((a, b) => 
    elementCounts[a] > elementCounts[b] ? a : b)
  
  const radius = getAtomRadius(mostCommonElement) * radiusScale
  const geometry = new THREE.IcosahedronGeometry(radius, 3)
  const material = new THREE.MeshPhysicalMaterial({
    color: getAtomColor(mostCommonElement),
    metalness: 0.0,
    roughness: 0.22,
    clearcoat: 1.0,
    clearcoatRoughness: 0.15,
    envMapIntensity: 0.5
  })
  
  const instancedMesh = new THREE.InstancedMesh(geometry, material, sites.length)
  
  // Set positions for each instance
  const matrix = new THREE.Matrix4()
  sites.forEach((site, index) => {
    if (site.cartesian) {
      matrix.setPosition(site.cartesian[0], site.cartesian[1], site.cartesian[2])
      instancedMesh.setMatrixAt(index, matrix)
    }
  })
  
  return instancedMesh
}