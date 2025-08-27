/**
 * JSON parser for custom crystal structure format
 */

import { CrystalStructure, fracToCartesian } from '../crystal'

export class JSONParser {
  /**
   * Parse JSON text content into a CrystalStructure object
   * @param jsonText The JSON file content as string
   * @returns Parsed CrystalStructure
   */
  public static parse(jsonText: string): CrystalStructure {
    const data = JSON.parse(jsonText)
    
    // Validate required fields
    if (!data.lattice || !data.lattice.matrix) {
      throw new Error('Missing lattice.matrix in JSON structure')
    }
    
    if (!data.sites || !Array.isArray(data.sites)) {
      throw new Error('Missing or invalid sites array in JSON structure')
    }
    
    // Convert fractional to cartesian coordinates if needed
    const latticeMatrix = data.lattice.matrix
    data.sites.forEach((site: any) => {
      if (site.frac && !site.cartesian) {
        site.cartesian = fracToCartesian(site.frac, latticeMatrix)
      }
    })
    
    // Ensure default values
    const crystal: CrystalStructure = {
      lattice: data.lattice,
      pbc: data.pbc || [true, true, true],
      sites: data.sites,
      site_properties: data.site_properties,
      style: data.style
    }
    
    return crystal
  }
  
  /**
   * Serialize a CrystalStructure object to JSON string
   * @param crystal The CrystalStructure to serialize
   * @returns JSON string representation
   */
  public static stringify(crystal: CrystalStructure): string {
    return JSON.stringify(crystal, null, 2)
  }
}