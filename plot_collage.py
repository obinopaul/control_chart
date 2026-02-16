
import os
from PIL import Image
import io
import base64
import webbrowser

def display_images_grid_scrollable_in_browser(folder_path, algorithms, abtypes, metric, img_width=300, img_height=300, output_html="output.html"):
    # Wrapping the table in a scrollable div with set width and horizontal scroll
    html = """
    <html><body>
    <div style='width: 100%; height: auto; overflow-x: auto; white-space: nowrap;'>
    <table style='border-spacing: 10px;'>
    """
    
    
    algorithm_name_map = {
    'PA1_Csplit': 'CSPA-I',
    'PA2_Csplit': 'CSPA-II',
    'OGD_1': 'CSOGD-I',
    'OGD_2': 'CSOGD-II',
    'PA1': 'PA-I',
    'PA2': 'PA-II',
    'PA1_L1': 'CSPA-I-L1',
    'PA2_L1': 'CSPA-II-L1',
    'PA1_L2': 'CSPA-I-L2',
    'PA2_L2': 'CSPA-II-L2',

    # New algorithms without C+ and C- 
    'PA_L1': 'PA-L1',
    'PA_L2': 'PA-L2',
    'PA_I_L1': 'PA-I-L1',
    'PA_I_L2': 'PA-I-L2',
    'PA_II_L1': 'PA-II-L1',
    'PA_II_L2': 'PA-II-L2'
    }
    
    
    # Add column headers (algorithm names)
    html += "<tr><td></td>"  # Empty cell for the abtype header column
    for algo in algorithms:
        html += f"<td style='font-weight: bold; text-align: center;'>{algorithm_name_map.get(algo, algo)}</td>"
    html += "</tr>"  # Close the header row
    
    # Loop through each abtype and algorithm to display the images
    for abtype in abtypes:
        html += f"<tr><td style='font-weight: bold; vertical-align: middle; white-space: nowrap'>{abtype}</td>"  # Row header for each abtype
        for algo in algorithms:
            img_path = os.path.join(folder_path, abtype, algo, f'{metric}.png')
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode("ascii")
                    # Display the image with fixed width and height in a scrollable table cell
                    html += f'<td><img src="data:image/png;base64,{img_base64}" style="width:{img_width}px; height:{img_height}px; border:1px solid black;"></td>'
            else:
                html += '<td><b>Image Not Found</b></td>'  # Placeholder if image not found
        html += "</tr>"  # End row
    html += "</table></div></body></html>"  # Close the table and the scrollable div
    
    # Save the generated HTML to a file
    with open(output_html, "w") as file:
        file.write(html)
    
    # Automatically open the HTML file in the browser
    webbrowser.open(f'file://{os.path.realpath(output_html)}')

# Example usage:
folder_path = 'plots_HeatMap'  # Parent folder containing abtype subfolders
algorithms = [ 'OGD', 'OGD_1', 'OGD_2', 'PA', 'PA1', 'PA1_Csplit', 'PA1_L1', 'PA1_L2', 'PA2', 'PA2_Csplit', 'PA2_L1', 'PA2_L2',
             'PA_L1', 'PA_L2', 'PA_I_L1', 'PA_I_L2', 'PA_II_L1', 'PA_II_L2']  # List of algorithms, you can add more here
abtypes = [f'abtype{i}' for i in range(1, 8)]  # abtype1, abtype2, etc.
metric = 'G-mean'  # Performance metric to display (e.g., G-means, Accuracies, etc.)

# Increase the image size by specifying img_width and img_height
display_images_grid_scrollable_in_browser(folder_path, algorithms, abtypes, metric, img_width=400, img_height=300)
