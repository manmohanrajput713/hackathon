"""Well types, per-well datasets and settings. The UI forms, templates and validation all read this file."""

# Column kinds: date (calendar day), time (UTC timestamp), number, text, choice (one of options).
# min/max are hard limits; values outside are rejected as physically invalid.
EVENT_TYPES = ["installation", "replacement", "planned_replacement", "valve_leak", "valve_malfunction",
               "pump_failure", "workover", "tubing_leak", "casing_heading", "inspection", "coverage_end", "other"]

DATASETS = {
    "esp_daily": {
        "label": "ESP daily sensors",
        "description": "One row per day of daily-average ESP sensor readings. Used by the ESP failure-risk model.",
        "key": "date",
        "columns": [
            {"key": "date", "label": "Date", "kind": "date", "required": True},
            {"key": "frequency_hz", "label": "Output frequency", "unit": "Hz", "kind": "number", "required": True, "min": 0, "max": 120},
            {"key": "discharge_psi", "label": "Discharge pressure", "unit": "psi", "kind": "number", "required": True, "min": 0},
            {"key": "intake_psi", "label": "Intake pressure", "unit": "psi", "kind": "number", "required": True, "min": 0},
            {"key": "pump_avg_psi", "label": "Pump average pressure", "unit": "psi", "kind": "number", "required": True, "min": 0},
            {"key": "motor_temp_f", "label": "Motor winding temperature", "unit": "°F", "kind": "number", "required": True},
            {"key": "esp_temp_f", "label": "ESP temperature", "unit": "°F", "kind": "number", "required": True},
            {"key": "current_a", "label": "Drive current", "unit": "A", "kind": "number", "required": True, "min": 0},
            {"key": "voltage_v", "label": "Drive voltage", "unit": "V", "kind": "number", "required": True, "min": 0},
        ],
        "chart": ["intake_psi", "discharge_psi"],
        "model": "ESP failure risk",
    },
    "esp_fast": {
        "label": "ESP high-frequency telemetry",
        "description": "Evenly spaced ESP operating data at the virtual flow meter's sampling interval "
                       "(set on the well). Pressures are absolute. Well tests are optional.",
        "key": "timestamp",
        "columns": [
            {"key": "timestamp", "label": "Timestamp (UTC)", "kind": "time", "required": True},
            {"key": "frequency_hz", "label": "Pump frequency", "unit": "Hz", "kind": "number", "required": True, "min": 0, "max": 120},
            {"key": "choke_percent", "label": "Choke opening", "unit": "%", "kind": "number", "required": True, "min": 0, "max": 100},
            {"key": "manifold_bar", "label": "Manifold pressure", "unit": "bar a", "kind": "number", "required": True, "min": 0},
            {"key": "reservoir_bar", "label": "Reservoir pressure", "unit": "bar a", "kind": "number", "required": True, "min": 0},
            {"key": "bottomhole_bar", "label": "Bottom-hole pressure", "unit": "bar a", "kind": "number", "required": True, "min": 0},
            {"key": "wellhead_bar", "label": "Wellhead pressure", "unit": "bar a", "kind": "number", "required": True, "min": 0},
            {"key": "test_m3h", "label": "Well test rate", "unit": "m³/h", "kind": "number", "required": False, "min": 0},
        ],
        "chart": ["bottomhole_bar", "wellhead_bar"],
        "model": "ESP virtual flow meter",
    },
    "gl_fast": {
        "label": "Gas-lift high-frequency telemetry",
        "description": "Valve positions and absolute pressures every 60 s (one gas_pinc control step) for the gas-lift virtual "
                       "flow meter. Casing and tubing-head pressure fix the plant state; a downhole gauge makes the estimate exact "
                       "through transients. Samples without pressures are carried forward from the previous one. Well tests are optional.",
        "key": "timestamp",
        "columns": [
            {"key": "timestamp", "label": "Timestamp (UTC)", "kind": "time", "required": True},
            {"key": "production_choke_percent", "label": "Production choke opening", "unit": "%", "kind": "number", "required": True, "min": 0, "max": 100},
            {"key": "gas_lift_choke_percent", "label": "Gas-lift choke opening", "unit": "%", "kind": "number", "required": True, "min": 0, "max": 100},
            {"key": "casing_bar", "label": "Casing pressure", "unit": "bar a", "kind": "number", "required": False, "min": 0},
            {"key": "tubing_head_bar", "label": "Tubing-head pressure", "unit": "bar a", "kind": "number", "required": False, "min": 0},
            {"key": "bottomhole_bar", "label": "Bottom-hole pressure", "unit": "bar a", "kind": "number", "required": False, "min": 0},
            {"key": "test_m3h", "label": "Well test rate", "unit": "m³/h", "kind": "number", "required": False, "min": 0},
        ],
        "chart": ["casing_bar", "tubing_head_bar"],
        "model": "Gas-lift virtual flow meter",
    },
    "gl_pressure": {
        "label": "Gas-lift pressures",
        "description": "Casing and tubing-head pressure readings (gauge). Used by the gas-lift valve failure model "
                       "and the casing-heading screen. Lift-gas rate here is artificial-lift gas, not reservoir injection.",
        "key": "timestamp",
        "columns": [
            {"key": "timestamp", "label": "Timestamp (UTC)", "kind": "time", "required": True},
            {"key": "casing_kpa", "label": "Casing pressure", "unit": "kPa g", "kind": "number", "required": True},
            {"key": "tubing_head_kpa", "label": "Tubing-head pressure", "unit": "kPa g", "kind": "number", "required": True},
            {"key": "lift_gas_sm3d", "label": "Lift-gas injection rate", "unit": "sm³/d", "kind": "number", "required": False, "min": 0},
            {"key": "choke_percent", "label": "Choke position", "unit": "%", "kind": "number", "required": False, "min": 0, "max": 100},
        ],
        "chart": ["casing_kpa", "tubing_head_kpa"],
        "model": "Gas-lift valve failure risk",
    },
    "gl_curve": {
        "label": "Tested gas-lift curve",
        "description": "Measured oil rate at each lift-gas injection rate (well tests). The optimizer uses the Well Models "
                       "curve when one exists and falls back to these tested points.",
        "key": "gas_sm3d",
        "columns": [
            {"key": "gas_sm3d", "label": "Lift-gas injection rate", "unit": "sm³/d", "kind": "number", "required": True, "min": 0},
            {"key": "oil_m3d", "label": "Oil rate", "unit": "m³/d", "kind": "number", "required": True, "min": 0},
        ],
        "chart": ["oil_m3d"],
        "model": "Optimizer (fallback curve)",
    },
    "events": {
        "label": "Maintenance events",
        "description": "Installations, replacements, failures and inspections. Gas-lift valve events feed the valve model's "
                       "service intervals. All events appear on the failure timeline.",
        "key": "timestamp",
        "columns": [
            {"key": "timestamp", "label": "Timestamp (UTC)", "kind": "time", "required": True},
            {"key": "event_type", "label": "Event type", "kind": "choice", "options": EVENT_TYPES, "required": True},
            {"key": "confirmed", "label": "Confirmed", "kind": "choice", "options": ["yes", "no"], "required": False},
            {"key": "event_id", "label": "Episode ID", "kind": "text", "required": False},
            {"key": "notes", "label": "Notes", "kind": "text", "required": False},
        ],
        "chart": [],
        "model": "Failure timeline",
    },
}

WELL_TYPES = {
    "esp": {"label": "ESP producer", "datasets": ["esp_daily", "esp_fast", "events"], "role": "producer", "lift": "esp"},
    "gas_lift": {"label": "Gas-lift producer", "datasets": ["gl_pressure", "gl_fast", "gl_curve", "events"], "role": "producer", "lift": "gas_lift"},
    "producer": {"label": "Flowing producer", "datasets": [], "role": "producer", "lift": "natural"},
    "injector": {"label": "Water injector", "datasets": [], "role": "injector", "lift": None},
}

STATUSES = ["active", "shut_in", "suspended", "abandoned"]

# Per-type settings. Settings without a default are optional and may be left blank.
SETTINGS = {
    "esp": [
        {"key": "current_hz", "group": "Operations", "label": "Current pump frequency", "unit": "Hz", "min": 0, "max": 120},
        {"key": "env_min_hz", "group": "Approved operating envelope", "label": "Minimum frequency", "unit": "Hz", "min": 0, "max": 120},
        {"key": "env_max_hz", "group": "Approved operating envelope", "label": "Maximum frequency", "unit": "Hz", "min": 0, "max": 120},
        {"key": "vfm_sample_seconds", "group": "Virtual flow meter", "label": "Sampling interval", "unit": "s", "default": 1, "min": 0.001,
         "help": "Spacing of the high-frequency telemetry the flow meter checkpoint was trained on."},
        {"key": "reservoir_bar", "group": "Well model", "label": "Reservoir pressure", "unit": "bar a", "default": 126, "min": 1},
        {"key": "pi_m3d_bar", "group": "Well model", "label": "Productivity index", "unit": "m³/d/bar", "default": 20.0, "min": 0.01},
        {"key": "density", "group": "Well model", "label": "Produced fluid density", "unit": "kg/m³", "default": 950, "min": 500, "max": 1300},
        {"key": "manifold_bar", "group": "Well model", "label": "Manifold pressure", "unit": "bar a", "default": 20, "min": 1},
        {"key": "choke_percent", "group": "Well model", "label": "Production choke opening", "unit": "%", "default": 70, "min": 1, "max": 100},
        {"key": "water_cut", "group": "Well model", "label": "Water cut", "unit": "%", "default": 30, "min": 0, "max": 100},
        {"key": "gor", "group": "Well model", "label": "Gas-oil ratio", "unit": "sm³/sm³", "default": 50, "min": 0},
        {"key": "pump_efficiency", "group": "Well model", "label": "Pump and motor efficiency", "unit": "%", "default": 55, "min": 5, "max": 100},
    ],
    "gas_lift": [
        {"key": "current_gas_sm3d", "group": "Operations", "label": "Current lift-gas rate", "unit": "sm³/d", "min": 0},
        {"key": "env_min_gas", "group": "Approved operating envelope", "label": "Minimum lift gas", "unit": "sm³/d", "min": 0},
        {"key": "env_max_gas", "group": "Approved operating envelope", "label": "Maximum lift gas", "unit": "sm³/d", "min": 0},
        {"key": "reservoir_bar", "group": "Well model", "label": "Reservoir pressure", "unit": "bar a", "default": 160, "min": 1},
        {"key": "pi_m3d_bar", "group": "Well model", "label": "Productivity index", "unit": "m³/d/bar", "default": 28.1, "min": 0.01},
        {"key": "density", "group": "Well model", "label": "Liquid density", "unit": "kg/m³", "default": 760, "min": 500, "max": 1300},
        {"key": "glr_mass", "group": "Well model", "label": "Formation gas/liquid mass ratio", "unit": "kg/kg", "default": 0, "min": 0, "max": 1},
        {"key": "water_cut", "group": "Well model", "label": "Water cut", "unit": "%", "default": 25, "min": 0, "max": 100},
        {"key": "tubing_depth_m", "group": "Well model", "label": "Tubing / injection depth", "unit": "m", "default": 2048, "min": 200, "max": 6000},
        {"key": "tubing_id_m", "group": "Well model", "label": "Tubing inner diameter", "unit": "m", "default": 0.134, "min": 0.03, "max": 0.3},
        {"key": "gas_supply_bar", "group": "Well model", "label": "Lift-gas supply pressure", "unit": "bar a", "default": 140, "min": 5},
        {"key": "choke_fraction", "group": "Well model", "label": "Production choke opening", "unit": "fraction", "default": 0.5, "min": 0.05, "max": 1},
    ],
}

# Facility limits and policy stored on each scenario. Blank limits mean "not constrained".
FACILITY = [
    {"key": "lift_gas_available_sm3d", "label": "Lift gas available", "unit": "sm³/d", "group": "Lift gas"},
    {"key": "compressor_capacity_sm3d", "label": "Compressor capacity", "unit": "sm³/d", "group": "Lift gas"},
    {"key": "compressor_kwh_per_1000sm3", "label": "Compression energy", "unit": "kWh per 1000 sm³", "group": "Lift gas", "default": 100},
    {"key": "separator_liquid_m3d", "label": "Separator liquid capacity", "unit": "m³/d", "group": "Separation"},
    {"key": "separator_gas_sm3d", "label": "Separator gas capacity", "unit": "sm³/d", "group": "Separation"},
    {"key": "esp_power_kw", "label": "Power available for ESPs", "unit": "kW", "group": "Power and emissions"},
    {"key": "emissions_kg_per_kwh", "label": "Grid emissions factor", "unit": "kg CO₂e/kWh", "group": "Power and emissions", "default": 0.4},
    {"key": "energy_weight", "label": "Energy penalty", "unit": "m³ oil per kWh", "group": "Objective", "default": 0},
    {"key": "emissions_weight", "label": "Emissions penalty", "unit": "m³ oil per kg CO₂e", "group": "Objective", "default": 0},
]

POLICIES = {
    "hold": "Wells in alert are held at their current setpoint",
    "derate": "Wells in alert may only reduce their setpoint",
    "ignore": "Risk is shown but does not restrict setpoints",
}


def public():
    return {"datasets": DATASETS, "wellTypes": WELL_TYPES, "statuses": STATUSES, "settings": SETTINGS,
            "facility": FACILITY, "policies": POLICIES}
