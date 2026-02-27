import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import io
import os
import re
import time
import xml.etree.ElementTree as ET
import streamlit.components.v1 as components

# For interactive visuals
try:
    from pyvis.network import Network
    PYVIS_OK = True
except ImportError:
    PYVIS_OK = False

# --- UTILS FOR XML ---
def extract_cdata(element):
    if element is not None and element.text:
        return element.text.strip()
    return ""

def generate_interactive_network(lineage_df, project_name):
    if not PYVIS_OK:
        return "<p>Pyvis library is required for interactive graphs. Please `pip install pyvis`.</p>"
        
    proj_df = lineage_df[lineage_df['Project'] == project_name]
    
    net = Network(height="500px", width="100%", bgcolor="transparent", font_color="white", directed=True)
    # Enable Physics for bouncy animation
    net.force_atlas_2based()
    
    added_nodes = set()
    
    def get_color_shape(ntype):
        ntype = str(ntype).lower()
        if 'job' in ntype: return '#e06666', 'box'
        if 'load' in ntype: return '#9fc5e8', 'database'
        if 'transform' in ntype: return '#b6d7a8', 'ellipse'
        if 'extract' in ntype: return '#ffe599', 'circle'
        if 'connection' in ntype: return '#f6b26b', 'box'
        return '#d9d9d9', 'dot'

    for _, row in proj_df.iterrows():
        p_name = row['Parent Name']
        c_name = row['Child Name']
        
        if not c_name: continue
            
        p_label = f"[{row['Parent Type']}]\n{p_name}"
        c_label = f"[{row['Child Type']}]\n{c_name}"
        
        p_col, p_shp = get_color_shape(row['Parent Type'])
        c_col, c_shp = get_color_shape(row['Child Type'])
        
        if p_name not in added_nodes:
            net.add_node(p_name, label=p_label, title=f"Type: {row['Parent Type']}", color=p_col, shape=p_shp)
            added_nodes.add(p_name)
            
        if c_name not in added_nodes:
            net.add_node(c_name, label=c_label, title=f"Type: {row['Child Type']}", color=c_col, shape=c_shp)
            added_nodes.add(c_name)
            
        net.add_edge(p_name, c_name, color="gray")

    import tempfile
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
    net.save_graph(path)
    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()
    return html

# --- PARSING FUNCTIONS FOR JEDOX ---

def parse_jedox_dynamic(html_content, filename):
    soup = BeautifulSoup(html_content, 'html.parser')
    extracted_rows = []
    
    cube_name = filename.replace(".htm", "").replace(".html", "")
    breadcrumb_btns = soup.find_all('a', class_=re.compile(r'x-breadcrumb-btn|sReportsBr_btn'))
    breadcrumb_texts = [btn.get_text(strip=True) for btn in breadcrumb_btns]
    if "Rules" in breadcrumb_texts:
        idx = breadcrumb_texts.index("Rules")
        if idx > 0: cube_name = breadcrumb_texts[idx-1]
    elif breadcrumb_texts:
        cube_name = breadcrumb_texts[-1]

    rows = soup.find_all('tr')
    id_idx, pos_idx, rule_idx = 0, 1, 2
    
    for row in rows:
        cells = row.find_all(['td', 'th'])
        cell_texts = [c.get_text(strip=True).lower() for c in cells]
        if "id" in cell_texts and "position" in cell_texts:
            id_idx = cell_texts.index("id")
            pos_idx = cell_texts.index("position")
            rule_idx = cell_texts.index("rule")
            break

    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 3: continue
            
        def get_clean_text(cell_idx):
            if cell_idx >= len(cells): return ""
            inner = cells[cell_idx].find('div')
            return inner.get_text(strip=True) if inner else cells[cell_idx].get_text(strip=True)

        def check_active_status():
            for cell in cells:
                span = cell.find('span', role='checkbox')
                if span:
                    return "Yes" if any('checked' in c for c in span.get('class', [])) else "No"
            return "No"

        try:
            rule_id_raw = get_clean_text(id_idx)
            position_raw = get_clean_text(pos_idx)
            if not position_raw.isdigit() and position_raw.lower() in ['position', '']: 
                continue

            extracted_rows.append({
                "ID": int(rule_id_raw) if rule_id_raw.isdigit() else rule_id_raw,
                "Position": int(position_raw) if position_raw.isdigit() else position_raw,
                "Rule": get_clean_text(rule_idx),
                "Comment": get_clean_text(5) if len(cells) > 5 else "",
                "Updated": get_clean_text(6) if len(cells) > 6 else "",
                "Active": check_active_status(),
                "Cube": cube_name,
                "SourceType": "HTML",
                "File Source": filename
            })
        except Exception:
            continue
            
    return extracted_rows

def parse_jds(content, filename):
    extracted_rows = []
    cube_match = re.search(r'VARIABLE_(?:DECLARE|DEFINE)\(CubeName\s*;\s*"(.*?)"', content, re.IGNORECASE)
    if not cube_match:
        cube_match = re.search(r'VARIABLE_(?:DECLARE|DEFINE)\([^;]+?Cube[^;]*?;\s*"(.*?)"', content, re.IGNORECASE)
    cube_name = cube_match.group(1) if cube_match else filename.replace(".jds", "")
    
    blocks = []
    start_idx = 0
    while True:
        idx = content.find('RULE_CREATE(', start_idx)
        if idx == -1:
            break
        
        i = idx + len('RULE_CREATE(')
        paren_count = 1
        in_quotes = False
        
        while i < len(content) and paren_count > 0:
            c = content[i]
            if c == '"':
                in_quotes = not in_quotes
            elif c == '(' and not in_quotes:
                paren_count += 1
            elif c == ')' and not in_quotes:
                paren_count -= 1
            i += 1
            
        blocks.append(content[idx + len('RULE_CREATE('):i-1])
        start_idx = i
    
    for idx, block in enumerate(blocks, start=1):
        params = []
        current = []
        in_quotes = False
        i = 0
        while i < len(block):
            c = block[i]
            if c == '"':
                if in_quotes and i + 1 < len(block) and block[i+1] == '"':
                    current.append('"'); i += 1
                else: in_quotes = not in_quotes
                current.append('"')
            elif c == ';' and not in_quotes:
                params.append("".join(current).strip()); current = []
            else: current.append(c)
            i += 1
        params.append("".join(current).strip())
        
        clean_params = [p[1:-1].replace('""', '"') if (p.startswith('"') and p.endswith('"')) else str(p) for p in params]
        
        if len(clean_params) < 5: continue
            
        rule_content = clean_params[1]
        active_val = clean_params[2]
        raw_id_param = clean_params[4]
        
        position_val = int(raw_id_param) if raw_id_param.isdigit() else idx

        comment_parts = []
        if len(clean_params) > 5:
             for j in range(5, len(clean_params)):
                p = clean_params[j]
                if p.strip() and p not in ["-1", "0"]: 
                    comment_parts.append(p.strip())
        
        extracted_rows.append({
            "Position": position_val, 
            "Rule": rule_content,
            "Comment": " | ".join(comment_parts),
            "Updated": "", 
            "Active": "Yes" if active_val == "1" else "No",
            "Cube": cube_name,
            "SourceType": "JDS",
            "File Source": filename
        })
        
    return extracted_rows

def parse_jedox_text(content, filename):
    extracted_rows = []
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    cube_name = filename.replace(".txt", "")
    
    if "Definition;Comment;Query" in content or "['" in content[:100]:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split(';')
            if len(parts) >= 5:
                rule_def = parts[0]
                active = parts[4] if len(parts) > 4 else ""
                timestamp = parts[6] if len(parts) > 6 else ""
                extern_id = parts[5] if len(parts) > 5 else ""
                
                pos_val = int(extern_id) if extern_id.isdigit() else i
                extracted_rows.append({
                    "Position": pos_val,
                    "Rule": rule_def,
                    "Comment": parts[1] if len(parts) > 1 else "",
                    "Updated": timestamp,
                    "Active": "Yes" if active == "1" else "No",
                    "Cube": cube_name,
                    "SourceType": "TXT",
                    "File Source": filename
                })
    return extracted_rows


def get_search_values(uploaded_file, manual_text):
    search_values = set()
    if manual_text:
        search_values.update([v.strip() for v in manual_text.split(',') if v.strip()])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_source = pd.read_csv(uploaded_file)
            else:
                df_source = pd.read_excel(uploaded_file, sheet_name=0)
            
            for col in df_source.columns:
                vals = df_source[col].dropna().astype(str).unique()
                search_values.update(v.strip() for v in vals if v.strip())
        except Exception as e:
            st.error(f"Error reading search file: {e}")
            
    return search_values

def parse_xml_keyword_search(content, filename, search_values):
    all_results = []
    
    tag_start_pattern = re.compile(r'<(project|transform|transformation|extract|load|job|connection)\s+[^>]*(?:name|nameref)="([^"]+)"', re.IGNORECASE)
    tag_end_pattern = re.compile(r'</(project|transform|transformation|extract|load|job|connection)>', re.IGNORECASE)
    
    context_stack = [] 
    project_name = "Unknown Project"
    
    lines = content.split('\n')
    for line_idx, line in enumerate(lines, 1):
        clean_line = line.strip()
        if not clean_line: continue
        
        start_match = tag_start_pattern.search(clean_line)
        just_opened_self_closing = False
        
        if start_match:
            tag_type = start_match.group(1).lower()
            tag_name = start_match.group(2)
            
            if tag_type == 'project':
                project_name = tag_name
            else:
                context_stack.append({"type": tag_type, "name": tag_name})
                
                if clean_line.rstrip().endswith("/>") or "/>" in clean_line[start_match.end()-2:start_match.end()+10]: 
                    end_of_tag_search = re.search(r'/?>', clean_line[start_match.start():])
                    if end_of_tag_search and end_of_tag_search.group().startswith('/'):
                        just_opened_self_closing = True
        
        found_vals = [val for val in search_values if val in clean_line]
        if found_vals:
            if context_stack:
                curr = context_stack[-1]
                c_type, c_name = curr["type"], curr["name"]
            else:
                c_type, c_name = "Global/Other", "N/A"
                
            all_results.append({
                "Project Name": project_name,
                "Source File": filename,
                "Line Number": line_idx,
                "Context Type": c_type,
                "Context Name": c_name,
                "Found Value": ", ".join(found_vals),
                "Line Content": clean_line[:1000]
            })
        
        end_match = tag_end_pattern.search(clean_line)
        if end_match:
            end_tag = end_match.group(1).lower()
            if end_tag != 'project':
                if context_stack and context_stack[-1]["type"] == end_tag:
                    context_stack.pop()

        if just_opened_self_closing:
            if context_stack and context_stack[-1]["type"] == tag_type:
                context_stack.pop()
                
    return all_results

def parse_xml_lineage(content, filename):
    data_variables = []
    data_connections = []
    data_extracts = []
    data_transforms = []
    data_loads = []
    data_jobs = []
    data_lineage = []
    node_details = {} # To hold specific code/queries for the interactive selector
    
    try:
        root = ET.fromstring(content)
        project_name = root.attrib.get('name', filename)
        node_types = {}
        
        # Variables
        for var in root.findall('.//variables/variable'):
            v_name = var.attrib.get('name', '')
            v_default = extract_cdata(var.find('default'))
            v_comment = extract_cdata(var.find('comment'))
            data_variables.append({
                "File": filename, "Project": project_name,
                "Variable Name": v_name, "Default Value": v_default, "Comment": v_comment
            })
            
        # Connections
        for conn in root.findall('.//connections/connection'):
            c_name = conn.attrib.get('name', '')
            c_type = conn.attrib.get('type', '')
            node_types[c_name] = 'Connection'
            
            details = []
            for child in conn:
                if child.tag != 'comment' and child.text:
                    details.append(f"{child.tag}={extract_cdata(child)}")
            
            data_connections.append({
                "File": filename, "Project": project_name,
                "Connection Name": c_name, "Type": c_type, "Details": " | ".join(details)
            })
            node_details[c_name] = {"Class": "Connection", "Props": "\n".join(details)}
            
        # Extracts
        for ext in root.findall('.//extracts/extract'):
            e_name = ext.attrib.get('name', '')
            e_type = ext.attrib.get('type', '')
            e_comment = extract_cdata(ext.find('comment'))
            node_types[e_name] = 'Extract'
            
            e_conn = ext.find('connection')
            e_conn_name = e_conn.attrib.get('nameref', '') if e_conn is not None else ""
            
            query_info = []
            query = ext.find('query')
            if query is not None:
                query_cube = query.find('cube')
                if query_cube is not None:
                    query_info.append(f"Cube: {query_cube.attrib.get('name', '')}")
                for dim in query.findall('.//dimensions/dimension'):
                    dim_name = dim.attrib.get('name', '')
                    conds = []
                    for cond in dim.findall('condition'):
                        op = cond.attrib.get('operator', '')
                        val = extract_cdata(cond)
                        conds.append(f"{op} '{val}'")
                    if conds:
                        query_info.append(f"Dim [{dim_name}]: {', '.join(conds)}")
                if query.text and query.text.strip():
                    query_info.append(f"SQL:\n{query.text.strip()}")
            
            if e_type == 'Relational' and ext.find('query') is None and ext.text:
               query_info.append(f"SQL:\n{ext.text.strip()}")
            
            data_extracts.append({
                "File": filename, "Project": project_name,
                "Extract Name": e_name, "Type": e_type, "Connection": e_conn_name, 
                "Details/Query": "\n".join(query_info), "Comment": e_comment
            })
            node_details[e_name] = {"Class": "Extract", "Props": "\n".join(query_info)}
            
            if e_conn_name:
                data_lineage.append({
                    "Project": project_name, "Parent Type": "Extract", "Parent Name": e_name,
                    "Child Type": "Connection", "Child Name": e_conn_name
                })
            
        # Transforms
        for tr in root.findall('.//transforms/transform'):
            t_name = tr.attrib.get('name', '')
            t_type = tr.attrib.get('type', '')
            t_comment = extract_cdata(tr.find('comment'))
            node_types[t_name] = 'Transform'
            
            sources = []
            for src in tr.findall('.//sources/source'):
                src_nameref = src.attrib.get('nameref', '')
                if src_nameref:
                    sources.append(src_nameref)
                    data_lineage.append({
                        "Project": project_name, "Parent Type": "Transform", "Parent Name": t_name,
                        "Child Type": "Pending", "Child Name": src_nameref
                    })
            
            details = []
            script = tr.find('script')
            if script is not None:
                details.append(f"Groovy Script:\n{extract_cdata(script)}\n")
            
            for coord in tr.findall('.//target/coordinates/coordinate'):
                c_name = coord.attrib.get('name', '')
                inp = coord.find('input')
                if inp is not None:
                    if 'nameref' in inp.attrib:
                        details.append(f"Map: {c_name} <- {inp.attrib['nameref']}")
                    elif 'constant' in inp.attrib:
                        details.append(f"Map: {c_name} <- const('{inp.attrib['constant']}')")
            
            for func in tr.findall('.//functions/function'):
                f_name = func.attrib.get('name', '')
                f_type = func.attrib.get('type', '')
                details.append(f"Function: {f_name} (Type: {f_type})")
                for finp in func.findall('input'):
                    ref = finp.attrib.get('nameref')
                    const = finp.attrib.get('constant')
                    arg_val = ref if ref else f"'{const}'"
                    details.append(f"  Arg -> {arg_val}")

            data_transforms.append({
                "File": filename, "Project": project_name,
                "Transform Name": t_name, "Type": t_type, "Sources": ", ".join(sources),
                "Details/Mapping": "\n".join(details), "Comment": t_comment
            })
            node_details[t_name] = {"Class": "Transform", "Props": "\n".join(details)}
            
        # Loads
        for ld in root.findall('.//loads/load'):
            l_name = ld.attrib.get('name', '')
            l_type = ld.attrib.get('type', '')
            l_comment = extract_cdata(ld.find('comment'))
            node_types[l_name] = 'Load'
            
            l_src = ld.find('source')
            l_src_name = l_src.attrib.get('nameref', '') if l_src is not None else ""
            
            l_conn = ld.find('connection')
            l_conn_name = l_conn.attrib.get('nameref', '') if l_conn is not None else ""
            
            l_mode = extract_cdata(ld.find('mode'))
            
            cube_details = []
            cube = ld.find('cube')
            if cube is not None:
                cube_details.append(f"Target Cube: {cube.attrib.get('name', '')}")
                for dim in cube.findall('.//dimensions/dimension'):
                    cube_details.append(f"  Dim: {dim.attrib.get('name', '')} <- Input: {dim.attrib.get('input', '')}")
                    
            rel_table = ld.find('table')
            if rel_table is not None:
                 cube_details.append(f"Target Table: {rel_table.attrib.get('name', '')}")
                 
            file_target = ld.find('file')
            if file_target is not None:
                 cube_details.append(f"Target File: {file_target.attrib.get('name', '')}")
            
            data_loads.append({
                "File": filename, "Project": project_name,
                "Load Name": l_name, "Type": l_type, "Source": l_src_name, "Connection": l_conn_name,
                "Mode": l_mode, "Details": "\n".join(cube_details), "Comment": l_comment
            })
            node_details[l_name] = {"Class": "Load", "Props": f"Mode: {l_mode}\n" + "\n".join(cube_details)}
            
            if l_src_name:
                data_lineage.append({
                    "Project": project_name, "Parent Type": "Load", "Parent Name": l_name,
                    "Child Type": "Pending", "Child Name": l_src_name
                })
            
        # Jobs
        for jb in root.findall('.//jobs/job'):
            j_name = jb.attrib.get('name', '')
            j_type = jb.attrib.get('type', '')
            j_comment = extract_cdata(jb.find('comment'))
            node_types[j_name] = 'Job'
            
            executions = []
            for i, ex in enumerate(jb.findall('.//executions/execution'), 1):
                ex_name = ex.attrib.get('nameref', '')
                if ex_name:
                    executions.append(f"{i}. {ex_name}")
                    data_lineage.append({
                        "Project": project_name, "Parent Type": "Job", "Parent Name": j_name,
                        "Child Type": "Pending", "Child Name": ex_name
                    })
                
            data_jobs.append({
                "File": filename, "Project": project_name,
                "Job Name": j_name, "Type": j_type, "Executions/Lineage": "\n".join(executions),
                "Comment": j_comment
            })
            node_details[j_name] = {"Class": "Job", "Props": "\n".join(executions)}
            
        for row in data_lineage:
            if row['Child Type'] == "Pending":
                child_name = row['Child Name']
                row['Child Type'] = node_types.get(child_name, 'Unknown Type')
                
    except Exception as e:
        print(f"Error parsing: {e}")
        
    return {
        "variables": data_variables, "connections": data_connections,
        "extracts": data_extracts, "transforms": data_transforms,
        "loads": data_loads, "jobs": data_jobs,
        "lineage": data_lineage, "node_details": node_details,
        "project": project_name if 'project_name' in locals() else filename
    }

# --- STREAMLIT UI ---
st.set_page_config(page_title="Data Extractor Hub", page_icon="📜", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    h1, h2, h3, h4, p, label { color: #ffffff !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #161b22; border-bottom: 1px solid #30363d; border-radius: 4px; }
    .stTabs [data-baseweb="tab"] { color: #8b949e !important; background-color: transparent !important; border: none !important; }
    .stTabs [aria-selected="true"] { color: #ffffff !important; border-bottom: 2px solid #58a6ff !important; }
    [data-testid="stFileUploader"] { background-color: #161b22; border: 1px dashed #30363d; border-radius: 8px; padding: 10px; }
    [data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
    [data-testid="stSidebar"] { background-color: #010409; border-right: 1px solid #30363d; }
    .node-box { background-color: #161b22; padding: 20px; border-radius: 8px; border: 1px solid #30363d; margin-top:20px;}
    .code-block { background-color: #0d1117; color: #c9d1d9; padding: 15px; border-radius: 5px; font-family: monospace; white-space: pre-wrap; word-wrap: break-word;}
</style>
""", unsafe_allow_html=True)

if 'master_data' not in st.session_state: st.session_state.master_data = []
if 'processed_files' not in st.session_state: st.session_state.processed_files = []
if 'xml_data' not in st.session_state: st.session_state.xml_data = []
if 'lineage_cache' not in st.session_state: st.session_state.lineage_cache = {}

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://www.jedox.com/wp-content/uploads/jedox-logo-header.png", width=120)
    st.title("Navigation")
    nav_option = st.radio("Select Tool", ["📜 Rule Registry Extractor", "🔍 XML Keyword Search", "⚙️ XML Lineage Visualizer"])
    
    st.divider()
    if st.button("🗑️ Reset All Cache"):
        st.session_state.master_data = []
        st.session_state.processed_files = []
        st.session_state.xml_data = []
        st.session_state.lineage_cache = {}
        st.rerun()

def add_data(file_data, name):
    st.session_state.master_data.extend(file_data)
    st.session_state.processed_files.append(name)

# ---------------------------------------------
# MODE 1: Rule Registry Extractor
# ---------------------------------------------
if nav_option == "📜 Rule Registry Extractor":
    st.title("📜 Jedox Rule Registry Extractor")

    rule_col1, rule_col2 = st.columns(2)
    with rule_col1:
        manual_keywords_rule = st.text_input("Manual Search Entry (comma separated)", key="rule_manual")
    with rule_col2:
        file_keywords_rule = st.file_uploader("Upload Keywords (CSV/Excel)", type=["csv", "xlsx", "xls"], key="rule_file")
        
    search_keywords_rule = get_search_values(file_keywords_rule, manual_keywords_rule)

    tab1, tab2, tab3 = st.tabs(["🚀 JDS Extraction", "📄 Text Extraction", "🌐 HTML Extraction"])

    with tab1:
        uploaded_jds = st.file_uploader("Upload .jds", type=['jds'], key="file_jds")
        if uploaded_jds and uploaded_jds.name not in st.session_state.processed_files:
            content = uploaded_jds.read().decode("utf-8", errors="ignore")
            file_data = parse_jds(content, uploaded_jds.name)
            if file_data:
                add_data(file_data, uploaded_jds.name)
                st.rerun()

    with tab2:
        uploaded_txt = st.file_uploader("Upload .txt", type=['txt'], key="file_txt")
        if uploaded_txt and uploaded_txt.name not in st.session_state.processed_files:
            content = uploaded_txt.read().decode("utf-8", errors="ignore")
            file_data = parse_jedox_text(content, uploaded_txt.name)
            if file_data:
                add_data(file_data, uploaded_txt.name)
                st.rerun()

    with tab3:
        uploaded_html = st.file_uploader("Maximum of 48 Rules can be extracted", type=['htm', 'html'], key="file_html")
        if uploaded_html and uploaded_html.name not in st.session_state.processed_files:
            st.session_state.master_data = [] 
            st.session_state.processed_files = []
            content = uploaded_html.read().decode("utf-8", errors="ignore")
            file_data = parse_jedox_dynamic(content, uploaded_html.name)
            if file_data:
                add_data(file_data, uploaded_html.name)
                st.rerun()

    if st.session_state.master_data:
        df = pd.DataFrame(st.session_state.master_data)
        for col in ['Position', 'ID']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

        if search_keywords_rule:
             def get_found_keywords(text):
                 if pd.isna(text): return ""
                 found = [kw for kw in search_keywords_rule if kw in str(text)]
                 return ", ".join(found)
             df['Found Value'] = df['Rule'].apply(get_found_keywords)
             df = df[df['Found Value'] != ""]

        if len(df) > 0:
            df = df.sort_values(by=["Cube", "Position"])
            cols = df.columns.tolist()
            primary_cols = [c for c in ['Position', 'ID', 'Rule', 'Found Value', 'Active', 'Cube', 'File Source'] if c in cols]
            df = df[primary_cols + [c for c in cols if c not in primary_cols]]

            st.dataframe(df, use_container_width=True, height=600)
            
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Rules')
            st.download_button("📥 Download Excel", buf.getvalue(), "Extracted_Rules.xlsx", key="dl_rules")

# ---------------------------------------------
# MODE 2: XML Keyword Search
# ---------------------------------------------
elif nav_option == "🔍 XML Keyword Search":
    st.title("🔍 XML Keyword Search")

    xml_col1, xml_col2 = st.columns(2)
    with xml_col1:
        manual_keywords_xml = st.text_input("Manual Search Entry (comma separated)", key="xml_manual")
    with xml_col2:
        file_keywords_xml = st.file_uploader("Upload Keywords File", type=["csv", "xlsx"], key="xml_file")

    search_keywords_xml = get_search_values(file_keywords_xml, manual_keywords_xml)
    uploaded_xmls = st.file_uploader("Upload .xml files", type=['xml'], accept_multiple_files=True, key="xml_targets")
    
    if st.button("🚀 Start XML Search", type="primary") and uploaded_xmls and search_keywords_xml:
        found_data = []
        for xml_file in uploaded_xmls:
            content = xml_file.read().decode("utf-8", errors="ignore")
            found_data.extend(parse_xml_keyword_search(content, xml_file.name, search_keywords_xml))
        st.session_state.xml_data = found_data
                    
    if st.session_state.xml_data:
        df_xml = pd.DataFrame(st.session_state.xml_data)
        st.dataframe(df_xml, use_container_width=True)
        buf_xml = io.BytesIO()
        with pd.ExcelWriter(buf_xml, engine='openpyxl') as writer:
            df_xml.to_excel(writer, index=False, sheet_name='Search Results')
        st.download_button("📥 Download Excel", buf_xml.getvalue(), "XML_Search_Report.xlsx", key="dl_xml")

# ---------------------------------------------
# MODE 3: XML Lineage Visualizer
# ---------------------------------------------
elif nav_option == "⚙️ XML Lineage Visualizer":
    st.title("⚙️ Jedox XML Lineage & Process Visualizer")
    
    uploaded_lineage_xmls = st.file_uploader("Upload Integrator XML files", type=['xml'], accept_multiple_files=True, key="lineage_files")
    
    if st.button("🔄 Process XMLs", type="primary"):
        if uploaded_lineage_xmls:
            all_data = {"variables": [], "connections": [], "extracts": [], "transforms": [], "loads": [], "jobs": [], "lineage": []}
            all_details = {}
            projects = []
            
            for xml_file in uploaded_lineage_xmls:
                content = xml_file.read().decode("utf-8", errors="ignore")
                parsed = parse_xml_lineage(content, xml_file.name)
                
                for k in ["variables", "connections", "extracts", "transforms", "loads", "jobs", "lineage"]:
                    all_data[k].extend(parsed[k])
                all_details.update(parsed["node_details"])
                projects.append(parsed["project"])
            
            st.session_state.lineage_cache = {
                "data": all_data,
                "details": all_details,
                "projects": sorted(list(set(projects)))
            }
            st.success(f"Processed {len(projects)} projects.")
        else:
            st.error("Upload XMLs first.")

    cache = st.session_state.lineage_cache
    if cache:
        df_lineage = pd.DataFrame(cache["data"]["lineage"])
        
        c_proj, c_btn = st.columns([3, 1])
        with c_proj:
            selected_proj = st.selectbox("📌 Select Project to Visualize", cache["projects"])
        
        tab_flow, tab_details, tab_tables = st.tabs(["🕸️ Interactive Data Flow", "🔍 Process Code & Details", "📊 Raw Data Tables & Export"])
        
        with tab_flow:
            st.info("Interactive Physics Network: You can drag nodes to rearrange them!")
            if not df_lineage.empty:
                html_code = generate_interactive_network(df_lineage, selected_proj)
                components.html(html_code, height=520)
            else:
                st.warning("No lineage mapping found.")

        with tab_details:
            st.write("### Inspect Specific Process Nodes")
            
            proj_nodes = []
            if not df_lineage.empty:
                pdf = df_lineage[df_lineage['Project'] == selected_proj]
                proj_nodes = sorted(list(set(pdf['Parent Name'].tolist() + pdf['Child Name'].tolist())))
            
            if proj_nodes:
                c_node1, c_node2 = st.columns([1, 2])
                with c_node1:
                    selected_node = st.selectbox("Select Node from Graph", proj_nodes)
                
                with c_node2:
                    if selected_node in cache["details"]:
                        node_info = cache["details"][selected_node]
                        st.markdown(f"<div class='node-box'><h4>🟦 Class: {node_info['Class']}</h4></div>", unsafe_allow_html=True)
                        st.write("**Configuration / Code:**")
                        st.markdown(f"<div class='code-block'>{node_info['Props']}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No detailed code block found for this node.")
                
            st.write("---")
            st.write("### 🧮 Project Global Variables")
            df_vars = pd.DataFrame(cache["data"]["variables"])
            if not df_vars.empty:
                proj_vars = df_vars[df_vars["Project"] == selected_proj]
                st.dataframe(proj_vars, use_container_width=True)

        with tab_tables:
            st.write("Review all parsed components directly.")
            buf_all = io.BytesIO()
            with pd.ExcelWriter(buf_all, engine='openpyxl') as writer:
                for sheet_name, sheet_key in [("Variables", "variables"), ("Connections", "connections"),
                                              ("Extracts", "extracts"), ("Transforms", "transforms"),
                                              ("Loads", "loads"), ("Jobs", "jobs"), ("Lineage", "lineage")]:
                    df_sheet = pd.DataFrame(cache["data"][sheet_key])
                    if not df_sheet.empty:
                        st.write(f"**{sheet_name}**")
                        st.dataframe(df_sheet, height=200, use_container_width=True)
                        df_sheet.to_excel(writer, index=False, sheet_name=sheet_name)
                    
            st.download_button("📥 Download Master Multi-File Excel Report", buf_all.getvalue(), "Jedox_Global_Extraction.xlsx", type="primary")