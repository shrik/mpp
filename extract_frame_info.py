import struct
import sys

def extract_frame_info(h264_file):
    with open(h264_file, 'rb') as f:
        data = f.read()

    frame_count = 0
    frame_types = {'I': 0, 'P': 0, 'B': 0}
    current_pos = 0

    while current_pos < len(data) - 4:
        # Look for start code
        if data[current_pos:current_pos+4] == b'\x00\x00\x00\x01':
            # Move past start code
            current_pos += 4
            
            # Get NAL unit type
            nal_unit_type = data[current_pos] & 0x1F
            
            if nal_unit_type in [1, 5]:  # Coded slice of a non-IDR or IDR picture
                frame_count += 1
                
                # Determine frame type
                first_mb_in_slice = struct.unpack('>B', data[current_pos+1:current_pos+2])[0]
                slice_type = (first_mb_in_slice & 0x1F) >> 2
                
                if nal_unit_type == 5 or slice_type == 2:
                    frame_types['I'] += 1
                elif slice_type == 0:
                    frame_types['P'] += 1
                elif slice_type == 1:
                    frame_types['B'] += 1
                
                # Extract user data and temporal ID
                metadata = extract_metadata(data, current_pos)
                for meta_type, meta_value in metadata:
                    if meta_type == 'user_data':
                        print(f"Frame {frame_count} User Data: {meta_value}")
                    elif meta_type == 'temporal_id':
                        print(f"Frame {frame_count} Temporal ID: {meta_value}")
        
        current_pos += 1

    print(f"Total frames: {frame_count}")
    print(f"I-frames: {frame_types['I']}")
    print(f"P-frames: {frame_types['P']}")
    print(f"B-frames: {frame_types['B']}")

def extract_metadata(data, start_pos):
    sei_start = start_pos
    metadata = []
    tmpdata = data
    # Search for SEI NAL unit (type 6)
    sei_start = tmpdata.find(b'\x00\x00\x00\x01', sei_start)
    
    if sei_start == -1:
        return metadata
    import pdb; pdb.set_trace()

    sei_start += 4  # Move past the start code and NAL unit type
    
    sei_size = 50

    user_data = tmpdata[sei_start+2:sei_start+2+sei_size].decode('utf-8', errors='ignore')
    metadata.append(('user_data', user_data.rstrip('\x00')))
    return metadata

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_frame_info.py <h264_file>")
        sys.exit(1)
    
    h264_file = sys.argv[1]
    extract_frame_info(h264_file)