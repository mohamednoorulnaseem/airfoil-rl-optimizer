"""
Production-Ready Dash Dashboard for Airfoil Optimization
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Import your modules
from src.aerodynamics.airfoil_gen import generate_naca_4digit
from src.aerodynamics.xfoil_interface import XFOILRunner
from src.validation.aircraft_comparison import AircraftComparator

# Initialize Dash app with professional theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    title="Airfoil Optimizer",
    update_title="Analyzing..."
)
server = app.server # Expose server for gunicorn/deployment

# Load trained RL model
try:
    rl_model = PPO.load("models/ppo_airfoil_final.zip")
    model_loaded = True
except:
    try:
        rl_model = PPO.load("models/ppo_airfoil_fake.zip")
        model_loaded = True
    except:
        model_loaded = False
        print("Warning: RL model not found. Manual mode only.")

# Initialize XFOIL and comparator
# We should be careful about creating multiple runners if they lock resources, but XFOILRunner seems to manage temp dirs.
xfoil_runner = XFOILRunner(reynolds=1e6, mach=0.0)
aircraft_comparator = AircraftComparator()

# ==================== LAYOUT ====================

app.layout = dbc.Container([
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1([
                html.I(className="fas fa-plane-departure me-3"),
                "RL + XFOIL Airfoil Optimizer"
            ], className="text-center text-primary my-4"),
            html.P(
                "Optimize NACA airfoils using Reinforcement Learning with real-time CFD validation",
                className="text-center text-muted mb-4"
            )
        ])
    ]),
    
    html.Hr(),
    
    # Main Content
    dbc.Row([
        
        # Left Panel: Controls
        dbc.Col([
            
            # Mode Selection
            dbc.Card([
                dbc.CardHeader(html.H5([
                    html.I(className="fas fa-cog me-2"),
                    "Optimization Mode"
                ])),
                dbc.CardBody([
                    dbc.RadioItems(
                        id='mode-selector',
                        options=[
                            {'label': 'Manual Tuning', 'value': 'manual'},
                            {'label': 'RL Optimization', 'value': 'rl', 
                             'disabled': not model_loaded}
                        ],
                        value='manual',
                        inline=True
                    )
                ])
            ], className="mb-3"),
            
            # Manual Controls
            dbc.Card([
                dbc.CardHeader(html.H5([
                    html.I(className="fas fa-sliders-h me-2"),
                    "Airfoil Parameters"
                ])),
                dbc.CardBody([
                    
                    # Max Camber (m)
                    html.Label("Max Camber (m)", className="fw-bold"),
                    dcc.Slider(
                        id='m-slider',
                        min=0.00, max=0.06, step=0.001,
                        value=0.02,
                        marks={i/100: f'{i/100:.2f}' for i in range(0, 7, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Br(),
                    
                    # Camber Position (p)
                    html.Label("Camber Position (p)", className="fw-bold"),
                    dcc.Slider(
                        id='p-slider',
                        min=0.10, max=0.70, step=0.01,
                        value=0.40,
                        marks={i/10: f'{i/10:.1f}' for i in range(1, 8, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Br(),
                    
                    # Thickness (t)
                    html.Label("Thickness (t)", className="fw-bold"),
                    dcc.Slider(
                        id='t-slider',
                        min=0.11, max=0.18, step=0.001,
                        value=0.12,
                        marks={i/100: f'{i/100:.2f}' for i in range(11, 19, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Br(),
                    
                    # Analyze Button
                    dbc.Button(
                        [html.I(className="fas fa-play me-2"), "Analyze with XFOIL"],
                        id='analyze-btn',
                        color="primary",
                        className="w-100 mb-2"
                    ),
                    
                    # RL Optimize Button
                    dbc.Button(
                        [html.I(className="fas fa-robot me-2"), "RL Optimize"],
                        id='rl-optimize-btn',
                        color="success",
                        className="w-100",
                        disabled=not model_loaded
                    ),
                ])
            ], className="mb-3"),
            
            # Aircraft Comparison
            dbc.Card([
                dbc.CardHeader(html.H5([
                    html.I(className="fas fa-plane me-2"),
                    "Aircraft Comparison"
                ])),
                dbc.CardBody([
                    dbc.Select(
                        id='aircraft-selector',
                        options=[
                            {'label': 'Boeing 737-800', 'value': 'Boeing 737-800'},
                            {'label': 'Airbus A320neo', 'value': 'Airbus A320neo'},
                            {'label': 'Boeing 787-8', 'value': 'Boeing 787-8'}
                        ],
                        value='Boeing 737-800'
                    ),
                    html.Br(),
                    dbc.Button(
                        [html.I(className="fas fa-chart-line me-2"), "Compare"],
                        id='compare-btn',
                        color="info",
                        className="w-100"
                    )
                ])
            ], className="mb-3"),
            
            # Loading Indicator
            dbc.Spinner(
                html.Div(id='loading-output'),
                color="primary",
                type="border"
            )
            
        ], width=3),
        
        # Right Panel: Visualizations
        dbc.Col([
            
            # Tabs for different views
            dbc.Tabs([
                
                # Tab 1: Airfoil & Performance
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='airfoil-plot', style={'height': '400px'})
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='performance-plot', style={'height': '400px'})
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='polar-plot', style={'height': '400px'})
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='ld-plot', style={'height': '400px'})
                        ], width=6)
                    ])
                ], label="Aerodynamics", tab_id="aero"),
                
                # Tab 2: Boeing Comparison
                dbc.Tab([
                    html.Div(id='comparison-content')
                ], label="Aircraft Comparison", tab_id="comparison"),
                
                # Tab 3: Metrics
                dbc.Tab([
                    html.Div(id='metrics-content')
                ], label="Performance Metrics", tab_id="metrics")
                
            ], id='tabs', active_tab='aero')
            
        ], width=9)
    ])
    
], fluid=True, className="p-4")


# ==================== CALLBACKS ====================

@app.callback(
    [Output('airfoil-plot', 'figure'),
     Output('performance-plot', 'figure'),
     Output('polar-plot', 'figure'),
     Output('ld-plot', 'figure'),
     Output('loading-output', 'children'),
     Output('m-slider', 'value'), # Update sliders after RL
     Output('p-slider', 'value'),
     Output('t-slider', 'value')],
    [Input('analyze-btn', 'n_clicks'),
     Input('rl-optimize-btn', 'n_clicks')],
    [State('m-slider', 'value'),
     State('p-slider', 'value'),
     State('t-slider', 'value'),
     State('mode-selector', 'value')]
)
def update_analysis(analyze_clicks, rl_clicks, m, p, t, mode):
    """Main analysis callback"""
    
    ctx = dash.callback_context
    if not ctx.triggered:
        # Check if sliders are None (initial load)
        if m is None: m = 0.02
        if p is None: p = 0.4
        if t is None: t = 0.12
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'rl-optimize-btn' and model_loaded:
            # Use RL to optimize
            m, p, t = optimize_with_rl()
            # Ensure they are python floats
            m, p, t = float(m), float(p), float(t)
    
    # Generate airfoil
    coords = generate_naca_4digit(m, p, t, n_points=100)
    
    # Run XFOIL analysis
    alpha_range = [0, 2, 4, 6, 8, 10]
    results = xfoil_runner.analyze_airfoil(coords, alpha_range=alpha_range)
    
    empty_fig = go.Figure()
    empty_common_outputs = [empty_fig, empty_fig, empty_fig, empty_fig, "Analysis failed", m, p, t]
    
    if not results:
        # XFOIL failed
        empty_fig.add_annotation(
            text="XFOIL analysis failed. Try different parameters.",
            showarrow=False,
            font=dict(size=16, color="red")
        )
        return empty_common_outputs
    
    # Create figures
    airfoil_fig = create_airfoil_plot(coords, m, p, t)
    perf_fig = create_performance_plot(results)
    polar_fig = create_polar_plot(results)
    ld_fig = create_ld_plot(results)
    
    loading_msg = f"✓ Analysis complete (m={m:.3f}, p={p:.3f}, t={t:.3f})"
    
    return airfoil_fig, perf_fig, polar_fig, ld_fig, loading_msg, m, p, t


@app.callback(
    Output('comparison-content', 'children'),
    [Input('compare-btn', 'n_clicks')],
    [State('m-slider', 'value'),
     State('p-slider', 'value'),
     State('t-slider', 'value'),
     State('aircraft-selector', 'value')]
)
def update_comparison(n_clicks, m, p, t, aircraft):
    """Aircraft comparison callback"""
    
    if not n_clicks:
        return html.Div("Click 'Compare' to see results", className="text-center text-muted p-5")
    
    if m is None: m = 0.02
    if p is None: p = 0.4
    if t is None: t = 0.12
    
    # Run comparison
    optimized_params = [m, p, t]
    comparison_result = aircraft_comparator.compare_to_aircraft(optimized_params, aircraft)
    
    if not comparison_result:
        return html.Div("Comparison failed", className="text-center text-danger p-5")
    
    # Create comparison visualization
    improvements = comparison_result['improvements']
    savings = comparison_result['fuel_savings']
    
    content = dbc.Container([
        
        # Summary Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"+{improvements['ld_improvement_percent']:.1f}%", 
                               className="text-success"),
                        html.P("L/D Improvement")
                    ])
                ], className="text-center")
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{improvements['cd_reduction_percent']:.1f}%", 
                               className="text-success"),
                        html.P("Drag Reduction")
                    ])
                ], className="text-center")
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"${savings['annual_cost_savings_usd']:,.0f}", 
                               className="text-primary"),
                        html.P("Annual Savings")
                    ])
                ], className="text-center")
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"${savings['fleet_lifetime_savings_usd']/1e9:.2f}B", 
                               className="text-danger"),
                        html.P("Fleet Savings (25yr)")
                    ])
                ], className="text-center")
            ], width=3)
        ], className="mb-4"),
        
        # Detailed comparison plot
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=create_comparison_plot(comparison_result))
            ])
        ])
        
    ])
    
    return content


@app.callback(
    Output('metrics-content', 'children'),
    [Input('analyze-btn', 'n_clicks')],
    [State('m-slider', 'value'),
     State('p-slider', 'value'),
     State('t-slider', 'value')]
)
def update_metrics(n_clicks, m, p, t):
    """Update metrics table"""
    
    if n_clicks is None:
         # Optionally show default metrics or nothing
         pass
    
    if m is None: m = 0.02
    if p is None: p = 0.4
    if t is None: t = 0.12

    # Generate and analyze
    coords = generate_naca_4digit(m, p, t, n_points=100)
    results = xfoil_runner.analyze_airfoil(coords, alpha_range=[0, 2, 4, 6, 8, 10])
    
    if not results:
        return html.Div("Analysis failed", className="text-center text-danger p-5")
    
    # Create metrics table
    df = pd.DataFrame(results)
    df['L/D'] = df['cl'] / df['cd']
    
    table = dbc.Table.from_dataframe(
        df[['alpha', 'cl', 'cd', 'L/D', 'cm']].round(4),
        striped=True,
        bordered=True,
        hover=True,
        responsive=True
    )
    
    return dbc.Container([
        html.H4("XFOIL Analysis Results", className="mb-3"),
        table,
        html.Hr(),
        html.H5("Summary Statistics"),
        html.P([
            html.Strong("Max L/D: "),
            f"{df['L/D'].max():.2f} at α={df.loc[df['L/D'].idxmax(), 'alpha']:.1f}°"
        ]),
        html.P([
            html.Strong("Max Cl: "),
            f"{df['cl'].max():.3f} at α={df.loc[df['cl'].idxmax(), 'alpha']:.1f}°"
        ])
    ])


# ==================== HELPER FUNCTIONS ====================

def optimize_with_rl():
    """Use RL model to find optimal parameters"""
    from src.optimization.airfoil_env import AirfoilEnvXFOIL
    
    env = AirfoilEnvXFOIL(use_xfoil=False) # Use surrogate for speed in prediction loop
    obs, _ = env.reset()
    
    for _ in range(50):  # 50 optimization steps
        action, _ = rl_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    
    return env.current_params


def create_airfoil_plot(coords, m, p, t):
    """Create airfoil geometry plot"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode='lines',
        line=dict(color='blue', width=3),
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.1)',
        name='Airfoil'
    ))
    
    fig.update_layout(
        title=f'NACA {int(m*100)}{int(p*10)}{int(t*100)} Airfoil',
        xaxis_title='x/c',
        yaxis_title='y/c',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        hovermode='closest',
        plot_bgcolor='white',
        height=400
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    
    return fig


def create_performance_plot(results):
    """Create Cl and Cd vs alpha plot"""
    
    alphas = [r['alpha'] for r in results]
    cls = [r['cl'] for r in results]
    cds = [r['cd'] for r in results]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=alphas, y=cls, name='Cl', 
                  line=dict(color='green', width=2),
                  mode='lines+markers'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=alphas, y=cds, name='Cd',
                  line=dict(color='red', width=2),
                  mode='lines+markers'),
        secondary_y=True
    )
    
    fig.update_layout(
        title='Lift & Drag Coefficients',
        xaxis_title='Angle of Attack (deg)',
        hovermode='x unified',
        height=400
    )
    
    fig.update_yaxes(title_text="Lift Coefficient (Cl)", secondary_y=False)
    fig.update_yaxes(title_text="Drag Coefficient (Cd)", secondary_y=True)
    
    return fig


def create_polar_plot(results):
    """Create polar curve (Cl vs Cd)"""
    
    cds = [r['cd'] for r in results]
    cls = [r['cl'] for r in results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cds, y=cls,
        mode='lines+markers',
        line=dict(color='purple', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Polar Curve',
        xaxis_title='Drag Coefficient (Cd)',
        yaxis_title='Lift Coefficient (Cl)',
        hovermode='closest',
        height=400
    )
    
    return fig


def create_ld_plot(results):
    """Create L/D vs alpha plot"""
    
    alphas = [r['alpha'] for r in results]
    lds = [r['cl']/r['cd'] for r in results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=alphas, y=lds,
        mode='lines+markers',
        line=dict(color='orange', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(255, 165, 0, 0.1)'
    ))
    
    fig.update_layout(
        title='Aerodynamic Efficiency',
        xaxis_title='Angle of Attack (deg)',
        yaxis_title='L/D Ratio',
        hovermode='closest',
        height=400
    )
    
    # Add max L/D annotation
    max_ld = max(lds)
    max_alpha = alphas[lds.index(max_ld)]
    
    fig.add_annotation(
        x=max_alpha, y=max_ld,
        text=f'Max L/D = {max_ld:.1f}',
        showarrow=True,
        arrowhead=2,
        bgcolor='yellow',
        bordercolor='black'
    )
    
    return fig


def create_comparison_plot(comparison_result):
    """Create Boeing comparison plot"""
    
    improvements = comparison_result['improvements']
    
    categories = ['L/D', 'Cl', 'Cd Reduction']
    values = [
        improvements['ld_improvement_percent'],
        improvements['cl_change_percent'],
        improvements['cd_reduction_percent']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=['green', 'blue', 'red'],
        text=[f'+{v:.1f}%' for v in values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Performance Improvements vs Boeing 737-800',
        yaxis_title='Improvement (%)',
        showlegend=False,
        height=400
    )
    
    return fig


# ==================== RUN APP ====================

if __name__ == '__main__':
    app.run(debug=True, port=8050)
