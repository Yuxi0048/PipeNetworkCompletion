import folium
import math
import pyproj
from shapely.geometry import LineString, MultiLineString
def utm_converter(linestring):
    line_coords = linestring.coords
    # Define the WGS84 and UTM coordinate systems
    wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 coordinate system
    utm = pyproj.CRS('EPSG:32633')   # UTM Zone 33N, for example (change to your relevant UTM zone)

    # Create a PyProj transformer for WGS84 to UTM conversion
    transformer = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True)

    # Convert WGS84 coordinates to UTM
    utm_coords = [transformer.transform(lon, lat) for lon, lat in line_coords]
    line = LineString(utm_coords)
    return line

def rotCalc(linestring):
    # Get the line's start and end coordinates
    start_coord, end_coord = linestring.coords[0], linestring.coords[-1]

    # Calculate the change in coordinates to determine the angle
    delta_x = end_coord[0] - start_coord[0]
    delta_y = end_coord[1] - start_coord[1]

    # Calculate the rotation angle in radians using arctan2
    rotation_angle_rad = math.atan2(delta_y, delta_x)

    # Convert radians to degrees
    rotation_angle_deg = math.degrees(rotation_angle_rad)
    return rotation_angle_deg

# Sample list of LineString objects
linestrings = gdf_ML1.geometry.values.tolist()
first_pt = [linestrings[0].coords[0][1], linestrings[0].coords[0][0]]
m = folium.Map(location=first_pt, zoom_start=100, tiles='openstreetmap')
# Add nodes and edges from LineString objects
for linestring in linestrings:
    if isinstance(linestring, MultiLineString):
        linestring = LineString([p for line in linestring.geoms for p in line.coords])
    coords = list(linestring.coords)
    rot = rotCalc(utm_converter(linestring))
    for i in range(len(coords)):
        node = coords[i]
        folium.RegularPolygonMarker(location=[node[1], node[0]], fill_color='blue', number_of_sides=3, radius=10, rotation=rot).add_to(m)       
    folium.PolyLine(locations=np.array(coords)[:,[1,0]], color='yellow').add_to(m)
m.save('graph_map.html')  # Save the map as an HTML file