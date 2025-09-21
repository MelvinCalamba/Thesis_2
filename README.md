# Thesis_2
This data contains a road data from Los Banos, Laguna. Latitude, Longitude, Road Class, Road Types will you encounter in this data. I hope it help this data for you project :>

start_node = The starting point of the road or like a entrance in a coordinate form (Latitude, Longitude).
end_node = The end point of the road like a exit in a coordinate form (Latitue, Longitude).
street_name = Literal na pangalan ng street.
road_direction = One Way or Two Way road.
distance_meters = Kung gaano kalayo ang start_node mo to end_node but in METER.
speed_limit_kph = Imaginary lang na speed limit (Naka set lang yung data namin sa 25-35 kph)
duration_seconds = Kung gaano kabilis makapunta ang sasakyan mula start_node to end_node in SECONDS (Equation: distance_meters * 3600 / (speed_limit_kph * 1000))
road_class = kung National road ba sya, Secondary road like Jamboree road, Maahas road, etc., Tertiary like mga baranggay roads, Campus road katulad ng mga roads sa UPLB, at Mountain road (mga road na madadaanan ang bundok)
road_type = Kung asphalt, uneven(medyo baku-bakong daan), dirt road, gravel road.
is_blocked = naka set lang sa zero ang laman ng dataset namin dahil ang simulation or system namin ang gagawa ng paraan sa column ng dataset na ito.
is_dead_end = (0 = hindi dead end, 1 = dead end).
