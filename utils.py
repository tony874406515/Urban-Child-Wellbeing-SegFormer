import numpy as np

# Label Mapping based on Table 1 in the manuscript
# Mapping ADE20K classes to 10 Research Labels
LABEL_MAPPING = {
    0: "Greenery",                # Originally: tree, grass, plant, bush
    1: "Building_facade",         # Originally: building, wall, fence
    2: "Pedestrian_cycleway",     # Originally: sidewalk, path, bike lane
    3: "Motorway",                # Originally: road, car, bus
    4: "Street_furniture",        # Originally: bench, pole, barricade, mailbox
    5: "Lighting_facilities",     # Originally: streetlight, street lamp
    6: "Surveillance_equipment",  # Originally: sign, monitor (Manual Correction applied)
    7: "Traffic_signs",           # Originally: traffic light, signal
    8: "Pedestrians_bicycles",    # Originally: person, rider, bicycle
    9: "Windows"                  # Originally: window, glass
}

def map_ade20k_to_research_labels(mask_array):
    """
    Converts standard ADE20K segmentation masks to the 10 classes used in this study.
    Note: This is a simplified mapping function. Actual mapping involves 
    complex index grouping based on the ADE20K index list.
    """
    # Placeholder logic for demonstration
    # In practice, you would map specific integer IDs from ADE20K to 0-9
    return mask_array
