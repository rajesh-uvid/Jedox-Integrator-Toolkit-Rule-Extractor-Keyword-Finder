Overview
The Jedox Integrator Toolkit is a comprehensive, Streamlit-based utility suite designed for Jedox developers and administrators. It automates the extraction and documentation of Jedox rules and ETL (Extract, Transform, Load) configurations, turning complex, interconnected scripts and XML files into readable formats and interactive diagrams.

Key Features
📜 Rule Registry Extractor
Extracts complete rule definitions from Jedox 

.jds
, .html, and Export 

.txt
 files.
Automatically parses parameters, IDs, comment blocks, active status, and timestamps.
Features keyword filtering (via manual entry or CSV/Excel lists) to quickly isolate specific rules.
Exports consolidated rule ledgers to structured Excel (.xlsx) files.
🔍 XML Keyword Search & Context Extractor
Performs deep, multi-file searches across Jedox Integrator XML project files.
Automatically identifies the context of the search hit (e.g., whether the keyword was found inside a Job, Transform, Load, or Extract).
Generates detailed, tabular context reports containing precise file names, line numbers, and code snippets.
⚙️ XML Lineage & Process Visualizer
Parses entire Jedox Integrator projects to build an intelligent parent-child data lineage map.
Interactive Flow Graphs: Uses physics-based network mapping (PyVis) to generate draggable, interactive visual flows of the ETL architecture (Job -> Load -> Transform -> Extract -> Connection).
Code Inspector: Instantly displays specific logic (Groovy scripts, SQL queries, mapping constants) for any node selected on the graph.
Master Export: Dumps all variables, scripts, and node dependencies across all uploaded projects into a multi-tabbed Excel workbook.
Technology Stack
Frontend: Streamlit
Data Processing: Pandas, BeautifulSoup4
Visualization: PyVis, NetworkX