import json

# Function to extract segment boundaries
def extract_segment_boundaries(data):
    for video_id, video_data in data.items():
        # Extract start points from scene_segments, excluding the last [last_frame, last_frame]
        segment_boundaries = [segment[0] for segment in video_data['scene_segments'][:-1]]
        segment_boundaries.append(video_data['scene_segments'][-1][0])
        # Add the boundaries list to the video data
        video_data['segment_boundaries'] = segment_boundaries
    return data

with open('dataset/activitynet/test_scene.json') as f:
    data = json.load(f)
# Process the data
updated_data = extract_segment_boundaries(data)

# Convert updated data back to JSON format
with open('dataset/activitynet/test_scene_boundary', 'w') as f:
    json.dump(data, f, indent=4)
