# Field Console

A Next.js dashboard for well performance, equipment health, and gas-lift allocation. The Rust API connects the dashboard to Python workers that run the production and maintenance models.

Start with the [setup, replay, and client data guide](field-console/README.md).

This repository contains source code. Before starting the historical replay from a fresh clone, generate the datasets and model outputs using the pipeline guides or supply your own local copies. The dashboard's offline preparation reads the CRM test CSVs, the ESP operating dataset and checkpoint, the processed pump dataset and training settings, the valve score export, and the gas-lift simulation output.

| Directory | Purpose |
| --- | --- |
| `field-console` | Dashboard, API, model integration, and Windows launcher |
| `pywaterflood` | Deep-CRM reservoir modeling |
| `pinn` | ESP virtual flow meter |
| `gas_pinc` | Gas-lift well physics and control simulation |
| `failure_1` | ESP maintenance forecasting |
| `failure_2` | Gas-lift valve fault forecasting |
