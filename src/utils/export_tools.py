"""
Industry Export Module - MATLAB and CAD Integration
Export airfoil data for SolidWorks, CATIA, MATLAB, etc.
"""

import numpy as np
import base64
import json

def get_export_buttons_html(m: float, p: float, t: float) -> str:
    """
    Generate HTML download buttons for Streamlit.
    Takes Naca parameters, generates coords, and creates data URI links.
    """
    try:
        from src.aerodynamics.airfoil_gen import naca4
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from src.aerodynamics.airfoil_gen import naca4

    xu, yu, xl, yl = naca4(m, p, t)
    
    # Coordinate formatting (Selig format)
    x_upper = xu[::-1]
    y_upper = yu[::-1]
    x_lower = xl[1:]
    y_lower = yl[1:]
    
    x = np.concatenate([x_upper, x_lower])
    y = np.concatenate([y_upper, y_lower])
    
    # DAT Format
    dat_content = "AIRFOIL\n"
    for xi, yi in zip(x, y):
        dat_content += f"{xi:.6f}  {yi:.6f}\n"
    b64_dat = base64.b64encode(dat_content.encode()).decode()
    
    # CSV Format
    csv_content = "x,y\n"
    for xi, yi in zip(x, y):
        csv_content += f"{xi:.6f},{yi:.6f}\n"
    b64_csv = base64.b64encode(csv_content.encode()).decode()
    
    # JSON Format
    json_content = json.dumps({
        "metadata": {"m": m, "p": p, "t": t},
        "coordinates": {"x": x.tolist(), "y": y.tolist()}
    }, indent=2)
    b64_json = base64.b64encode(json_content.encode()).decode()
    
    # Simple CSS for buttons
    style = """
    <style>
        .dwn-btn {
            display: inline-block;
            padding: 8px 16px;
            background-color: #2563EB;
            color: white !important;
            border-radius: 6px;
            text-decoration: none;
            margin-right: 10px;
            margin-bottom: 10px;
            font-family: sans-serif;
            font-weight: 600;
            transition: background-color 0.2s;
        }
        .dwn-btn:hover {
            background-color: #1D4ED8;
        }
    </style>
    """
    
    html = f"""
    {style}
    <div style="margin-top: 10px;">
        <a href="data:text/plain;base64,{b64_dat}" download="naca_{int(m*100)}{int(p*10)}{int(t*100)}.dat" class="dwn-btn">📄 Download .DAT</a>
        <a href="data:text/csv;base64,{b64_csv}" download="naca_{int(m*100)}{int(p*10)}{int(t*100)}.csv" class="dwn-btn">📊 Download .CSV</a>
        <a href="data:application/json;base64,{b64_json}" download="naca_{int(m*100)}{int(p*10)}{int(t*100)}.json" class="dwn-btn">🤖 Download .JSON</a>
    </div>
    """
    return html

# Legacy functions preserved for script compatibility if needed
def export_to_dat(coords: np.ndarray, filename: str = "airfoil.dat") -> str:
    with open(filename, 'w') as f:
        f.write("AIRFOIL\n")
        for x, y in coords:
            f.write(f"{x:.6f}  {y:.6f}\n")
    return filename
