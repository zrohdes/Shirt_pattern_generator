import streamlit as st
import cv2
import numpy as np
from PIL import Image
import ezdxf
import io
import zipfile
from datetime import datetime
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Shirt Pattern Generator",
    page_icon="👔",
    layout="wide"
)


def process_shirt_image(image):
    """
    Process the uploaded shirt image to extract features and measurements.
    This is a placeholder function - in a real implementation, you would use
    advanced computer vision techniques to detect shirt boundaries, seams, etc.
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_cv = img_array

    # Basic image processing (placeholder)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find contours (simplified approach)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour (assuming it's the shirt)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Extract measurements (in pixels, would need calibration for real measurements)
        # Converting to approximate real measurements in cm
        scale_factor = 0.1  # Approximate conversion from pixels to cm

        measurements = {
            'chest': round(w * 0.8 * scale_factor, 2),  # Chest width
            'waist': round(w * 0.7 * scale_factor, 2),  # Waist width
            'shoulder_width': round(w * 0.6 * scale_factor, 2),  # Shoulder width
            'back_length': round(h * 0.7 * scale_factor, 2),  # Back length
            'sleeve_length': round(h * 0.35 * scale_factor, 2),  # Sleeve length
            'neck_circumference': round(w * 0.15 * scale_factor, 2),  # Neck circumference
            'armhole_depth': round(h * 0.25 * scale_factor, 2),  # Armhole depth
            'cuff_circumference': round(w * 0.12 * scale_factor, 2),  # Cuff circumference
            'hem_width': round(w * 0.75 * scale_factor, 2)  # Hem width
        }

        return measurements, edges

    return None, edges


def generate_shirt_patterns(measurements, pattern_type="basic"):
    """
    Generate 2D cut patterns based on extracted measurements.
    This creates basic pattern pieces for a shirt.
    """
    patterns = {}

    if measurements:
        # Convert cm to mm for DXF (standard CAD units)
        chest = measurements['chest'] * 10
        waist = measurements['waist'] * 10
        shoulder_width = measurements['shoulder_width'] * 10
        back_length = measurements['back_length'] * 10
        sleeve_length = measurements['sleeve_length'] * 10
        armhole_depth = measurements['armhole_depth'] * 10
        hem_width = measurements['hem_width'] * 10

        # Front panel pattern
        patterns['front_panel'] = [
            (0, 0),
            (chest, 0),
            (chest, back_length),
            (0, back_length),
            (0, 0)
        ]

        # Back panel pattern
        patterns['back_panel'] = [
            (0, 0),
            (chest, 0),
            (chest, back_length),
            (0, back_length),
            (0, 0)
        ]

        # Sleeve pattern
        sleeve_width = shoulder_width * 0.8
        patterns['sleeve_left'] = [
            (0, 0),
            (sleeve_width, 0),
            (sleeve_width * 0.7, sleeve_length),
            (sleeve_width * 0.3, sleeve_length),
            (0, 0)
        ]

        patterns['sleeve_right'] = patterns['sleeve_left'].copy()

        # Collar pattern
        collar_width = measurements['neck_circumference'] * 10
        collar_height = 50  # Standard collar height in mm

        patterns['collar'] = [
            (0, 0),
            (collar_width, 0),
            (collar_width, collar_height),
            (0, collar_height),
            (0, 0)
        ]

    return patterns


def create_dxf_file(patterns):
    """
    Create a DXF file from the generated patterns.
    """
    # Create a new DXF document
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    # Add patterns to DXF with proper spacing
    x_offset = 0
    y_offset = 0
    max_width = 0

    for pattern_name, points in patterns.items():
        # Adjust points with current offset
        adjusted_points = [(x + x_offset, y + y_offset) for x, y in points]

        # Add pattern as polyline
        msp.add_lwpolyline(adjusted_points, close=True)

        # Add text label with proper positioning
        text_x = adjusted_points[0][0]
        text_y = adjusted_points[0][1] - 20

        msp.add_text(
            pattern_name.replace('_', ' ').title(),
            dxfattribs={
                'height': 10,
                'insert': (text_x, text_y)
            }
        )

        # Calculate pattern bounds for spacing
        x_coords = [x for x, y in adjusted_points]
        y_coords = [y for x, y in adjusted_points]
        pattern_width = max(x_coords) - min(x_coords)
        pattern_height = max(y_coords) - min(y_coords)

        # Update max width for row management
        max_width = max(max_width, pattern_width)

        # Arrange patterns in a grid layout
        if pattern_name in ['sleeve_right', 'collar']:
            x_offset += pattern_width + 50  # Place side by side
        else:
            x_offset = 0  # New row
            y_offset += pattern_height + 100  # Move to next row

    return doc


def main():
    st.title("👔 Shirt Pattern Generator")
    st.markdown("Upload a shirt image to generate 2D cut patterns in DXF format")

    # Sidebar for settings
    st.sidebar.header("Settings")
    pattern_type = st.sidebar.selectbox(
        "Pattern Type",
        ["Basic", "Advanced", "Custom"]
    )
    #
    scale_factor = st.sidebar.slider(
        "Scale Factor (pixels to mm)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1
    )

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Upload Shirt Image")
        uploaded_file = st.file_uploader(
            "Choose a shirt image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a shirt for pattern generation"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Shirt Image", use_column_width=True)

            # Process image button
            if st.button("Generate Patterns", type="primary"):
                with st.spinner("Processing image and generating patterns..."):
                    # Process the image
                    measurements, edges = process_shirt_image(image)

                    if measurements:
                        # Display measurements
                        st.subheader("Extracted Measurements")
                        st.json(measurements)

                        # Generate patterns
                        patterns = generate_shirt_patterns(measurements, pattern_type.lower())

                        # Create DXF file
                        dxf_doc = create_dxf_file(patterns)

                        # Save to bytes buffer instead of temporary file
                        dxf_buffer = io.BytesIO()
                        try:
                            # Create a temporary file to save DXF, then read it
                            with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp_file:
                                tmp_file_path = tmp_file.name

                            # Save DXF to temporary file
                            dxf_doc.saveas(tmp_file_path)

                            # Read the file content
                            with open(tmp_file_path, 'rb') as f:
                                dxf_content = f.read()

                        except Exception as e:
                            st.error(f"Error creating DXF file: {str(e)}")
                            dxf_content = None
                        finally:
                            # Clean up temporary file safely
                            try:
                                if os.path.exists(tmp_file_path):
                                    os.unlink(tmp_file_path)
                            except:
                                pass  # Ignore cleanup errors

                        # Store in session state only if DXF creation was successful
                        if dxf_content:
                            st.session_state['dxf_content'] = dxf_content
                            st.session_state['patterns'] = patterns
                            # measurements and edges already stored above
                            st.success("✅ Patterns generated successfully!")
                        else:
                            st.error("Failed to create DXF file. Please try again.")
                    else:
                        st.error("Could not detect shirt in the image. Please try with a clearer image.")

    with col2:
        st.header("Results")

        if 'measurements' in st.session_state:
            # Display edge detection result
            st.subheader("Edge Detection")
            if 'edges' in st.session_state:
                edges_image = Image.fromarray(st.session_state['edges'])
                st.image(edges_image, caption="Detected Edges", use_column_width=True)

            # Display pattern information
            st.subheader("Generated Patterns")
            patterns = st.session_state['patterns']

            for pattern_name, points in patterns.items():
                with st.expander(f"{pattern_name.replace('_', ' ').title()} Pattern"):
                    st.write(f"Number of points: {len(points)}")
                    st.write("Coordinates:")
                    for i, (x, y) in enumerate(points):
                        st.write(f"Point {i + 1}: ({x:.2f}, {y:.2f})")

            # Download button
            if 'dxf_content' in st.session_state:
                st.download_button(
                    label="📥 Download DXF File",
                    data=st.session_state['dxf_content'],
                    file_name=f"shirt_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dxf",
                    mime="application/dxf",
                    type="primary"
                )

                st.success("✅ Patterns generated successfully!")
        else:
            st.info("Upload a shirt image and click 'Generate Patterns' to see results here.")

    # Instructions
    # st.markdown("---")
    # st.subheader("📋 Instructions")
    # st.markdown("""
    # 1. **Upload Image**: Choose a clear, well-lit image of a shirt
    # 2. **Adjust Settings**: Modify pattern type and scale factor if needed
    # 3. **Generate**: Click the 'Generate Patterns' button to process the image
    # 4. **Download**: Download the generated DXF file containing cut patterns
    #
    # **Note**: This is a basic implementation. For production use, you would need:
    # - Advanced computer vision algorithms for accurate shirt detection
    # - Machine learning models trained on garment patterns
    # - Proper measurement calibration and scaling
    # - More sophisticated pattern generation algorithms
    # """)

    # # Technical notes
    # with st.expander("🔧 Technical Details"):
    #     st.markdown("""
    #     **Current Implementation:**
    #     - Uses OpenCV for basic edge detection
    #     - Generates simplified rectangular patterns
    #     - Creates DXF files using ezdxf library
    #     - Provides basic measurements extraction
    #
    #     **For Production Enhancement:**
    #     - Implement deep learning models for garment segmentation
    #     - Add support for different shirt types and styles
    #     - Include seam allowances and notches in patterns
    #     - Add measurement validation and correction tools
    #     - Support for different fabric types and stretch factors
    #     """)


if __name__ == "__main__":
    main()