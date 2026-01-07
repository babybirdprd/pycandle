// Report generation for PyCandle coverage analysis
use crate::LayerMeta;
use std::collections::HashMap;

/// Data structure holding analysis results
pub struct ReportData {
    pub supported: usize,
    pub unsupported: usize,
    pub gaps: HashMap<String, usize>,
    pub layers: HashMap<String, LayerMeta>,
}

/// Generates HTML coverage reports from manifest data
pub struct ReportGenerator {
    manifest: HashMap<String, LayerMeta>,
}

impl ReportGenerator {
    pub fn new(manifest: HashMap<String, LayerMeta>) -> Self {
        Self { manifest }
    }

    /// Analyze the manifest and categorize layers
    pub fn analyze(&self) -> ReportData {
        let mut supported = 0;
        let mut unsupported = 0;
        let mut gaps: HashMap<String, usize> = HashMap::new();

        for (_name, meta) in &self.manifest {
            if !meta.is_leaf {
                continue;
            }
            if self.is_supported(&meta.module_type) {
                supported += 1;
            } else {
                unsupported += 1;
                *gaps.entry(meta.module_type.clone()).or_default() += 1;
            }
        }

        ReportData {
            supported,
            unsupported,
            gaps,
            layers: self.manifest.clone(),
        }
    }

    /// Check if a module type is supported by PyCandle codegen
    fn is_supported(&self, module_type: &str) -> bool {
        matches!(
            module_type,
            "Linear"
                | "Conv1d"
                | "Conv2d"
                | "Embedding"
                | "LayerNorm"
                | "ReLU"
                | "GELU"
                | "Sigmoid"
                | "Tanh"
                | "ELU"
                | "LeakyReLU"
                | "Snake"
                | "BatchNorm1d"
                | "BatchNorm2d"
                | "LSTM"
        )
    }

    /// Generate a standalone HTML coverage report
    pub fn generate_html(&self, data: &ReportData) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>PyCandle Coverage Report</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #e2e8f0;
            --text-muted: #94a3b8;
            --green: #22c55e;
            --red: #ef4444;
            --blue: #3b82f6;
            --border: #334155;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 8px;
            background: linear-gradient(135deg, var(--blue), var(--green));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .subtitle {{
            color: var(--text-muted);
            margin-bottom: 32px;
        }}
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .card {{
            background: var(--card-bg);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--border);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        }}
        .card h3 {{
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }}
        .card .value {{
            font-size: 3rem;
            font-weight: 700;
        }}
        .card.supported .value {{ color: var(--green); }}
        .card.unsupported .value {{ color: var(--red); }}
        .card.total .value {{ color: var(--blue); }}
        .progress-bar {{
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--green), var(--blue));
            border-radius: 4px;
            transition: width 0.5s ease;
        }}
        h2 {{
            font-size: 1.5rem;
            font-weight: 600;
            margin: 32px 0 16px 0;
            color: var(--text);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 32px;
        }}
        th {{
            text-align: left;
            padding: 16px;
            background: rgba(0,0,0,0.2);
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}
        td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        tr:hover {{
            background: rgba(255,255,255,0.02);
        }}
        .status {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
        }}
        .status.supported {{
            background: rgba(34, 197, 94, 0.15);
            color: var(--green);
        }}
        .status.unsupported {{
            background: rgba(239, 68, 68, 0.15);
            color: var(--red);
        }}
        .status::before {{
            content: '';
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: currentColor;
        }}
        .mono {{
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.875rem;
        }}
        .shape {{
            color: var(--text-muted);
            font-size: 0.8rem;
        }}
        .count-badge {{
            display: inline-block;
            background: var(--border);
            padding: 2px 10px;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
        }}
        
        /* Component grouping styles */
        .component-section {{
            background: var(--card-bg);
            border-radius: 12px;
            border: 1px solid var(--border);
            margin-bottom: 16px;
            overflow: hidden;
        }}
        .component-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 20px;
            background: rgba(0,0,0,0.2);
            cursor: pointer;
            user-select: none;
            transition: background 0.2s;
        }}
        .component-header:hover {{
            background: rgba(0,0,0,0.3);
        }}
        .component-header h3 {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--text);
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .component-header .chevron {{
            transition: transform 0.2s;
            color: var(--text-muted);
        }}
        .component-header.collapsed .chevron {{
            transform: rotate(-90deg);
        }}
        .component-stats {{
            display: flex;
            gap: 16px;
            font-size: 0.875rem;
        }}
        .component-stats .stat {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .component-stats .stat.ok {{ color: var(--green); }}
        .component-stats .stat.err {{ color: var(--red); }}
        .component-content {{
            max-height: 2000px;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }}
        .component-content.collapsed {{
            max-height: 0;
        }}
        .component-content table {{
            margin-bottom: 0;
            border-radius: 0;
        }}
        .layer-name {{
            padding-left: 24px;
            position: relative;
        }}
        .layer-name::before {{
            content: '└';
            position: absolute;
            left: 8px;
            color: var(--border);
        }}
        
        /* Filters */
        .filters {{
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .filter-btn {{
            padding: 8px 16px;
            border: 1px solid var(--border);
            background: transparent;
            color: var(--text);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.875rem;
        }}
        .filter-btn:hover {{
            background: var(--card-bg);
        }}
        .filter-btn.active {{
            background: var(--blue);
            border-color: var(--blue);
        }}
        .search-box {{
            flex: 1;
            min-width: 200px;
            padding: 8px 16px;
            border: 1px solid var(--border);
            background: var(--card-bg);
            color: var(--text);
            border-radius: 8px;
            font-size: 0.875rem;
        }}
        .search-box:focus {{
            outline: none;
            border-color: var(--blue);
        }}
        .hidden {{ display: none !important; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🕯️ PyCandle Coverage Report</h1>
        <p class="subtitle">Module coverage analysis for Candle code generation</p>
        
        <div class="dashboard">
            <div class="card total">
                <h3>Total Layers</h3>
                <div class="value">{total}</div>
            </div>
            <div class="card supported">
                <h3>Supported</h3>
                <div class="value">{supported}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {coverage:.1}%"></div>
                </div>
            </div>
            <div class="card unsupported">
                <h3>Needs Implementation</h3>
                <div class="value">{unsupported}</div>
            </div>
        </div>
        
        <h2>Gap Analysis</h2>
        <table>
            <thead>
                <tr>
                    <th>Module Type</th>
                    <th>Count</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {gaps_table}
            </tbody>
        </table>
        
        <h2>Layers by Component</h2>
        <div class="filters">
            <input type="text" class="search-box" placeholder="Search layers..." id="searchBox">
            <button class="filter-btn active" data-filter="all">All</button>
            <button class="filter-btn" data-filter="supported">Supported Only</button>
            <button class="filter-btn" data-filter="unsupported">Unsupported Only</button>
            <button class="filter-btn" data-filter="expand">Expand All</button>
            <button class="filter-btn" data-filter="collapse">Collapse All</button>
        </div>
        
        {components_html}
    </div>
</body>
</html>"#,
            total = data.supported + data.unsupported,
            supported = data.supported,
            unsupported = data.unsupported,
            coverage = if data.supported + data.unsupported > 0 {
                (data.supported as f64 / (data.supported + data.unsupported) as f64) * 100.0
            } else {
                100.0
            },
            gaps_table = self.render_gaps_table(&data.gaps),
            components_html = self.render_components(&data.layers),
        )
    }

    fn render_gaps_table(&self, gaps: &HashMap<String, usize>) -> String {
        if gaps.is_empty() {
            return "<tr><td colspan=\"3\" style=\"text-align: center; color: var(--green);\">✅ All module types are supported!</td></tr>".to_string();
        }

        let mut sorted: Vec<_> = gaps.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        sorted
            .iter()
            .map(|(module_type, count)| {
                format!(
                    r#"<tr>
                        <td class="mono">{}</td>
                        <td><span class="count-badge">{}</span></td>
                        <td><span class="status unsupported">Needs Implementation</span></td>
                    </tr>"#,
                    module_type, count
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Group layers by their top-level component and render as collapsible sections
    fn render_components(&self, layers: &HashMap<String, LayerMeta>) -> String {
        // Group layers by their first path component
        let mut groups: HashMap<String, Vec<(&String, &LayerMeta)>> = HashMap::new();

        for (name, meta) in layers.iter().filter(|(_, m)| m.is_leaf) {
            let component = name.split('.').next().unwrap_or(name).to_string();
            groups.entry(component).or_default().push((name, meta));
        }

        // Sort groups by name
        let mut sorted_groups: Vec<_> = groups.into_iter().collect();
        sorted_groups.sort_by(|a, b| a.0.cmp(&b.0));

        let components_html = sorted_groups
            .iter()
            .map(|(component, layers)| {
                // Sort layers within component
                let mut sorted_layers = layers.clone();
                sorted_layers.sort_by(|a, b| a.0.cmp(b.0));

                // Count supported/unsupported
                let supported_count = sorted_layers
                    .iter()
                    .filter(|(_, m)| self.is_supported(&m.module_type))
                    .count();
                let unsupported_count = sorted_layers.len() - supported_count;

                let rows: String = sorted_layers
                    .iter()
                    .map(|(name, meta)| {
                        let supported = self.is_supported(&meta.module_type);
                        let status_class = if supported {
                            "supported"
                        } else {
                            "unsupported"
                        };
                        let status_text = if supported {
                            "Supported"
                        } else {
                            "Needs Implementation"
                        };

                        // Get the short name (everything after the first dot)
                        let short_name = name.split('.').skip(1).collect::<Vec<_>>().join(".");
                        let display_name = if short_name.is_empty() {
                            name.to_string()
                        } else {
                            short_name
                        };

                        let input_shapes = meta
                            .input_shapes
                            .iter()
                            .map(|s| format!("{:?}", s))
                            .collect::<Vec<_>>()
                            .join(", ");
                        let output_shapes = meta
                            .output_shapes
                            .iter()
                            .map(|s| format!("{:?}", s))
                            .collect::<Vec<_>>()
                            .join(", ");

                        format!(
                            r#"<tr class="layer-row" data-supported="{}">
                            <td class="mono layer-name">{}</td>
                            <td class="mono">{}</td>
                            <td class="shape">{}</td>
                            <td class="shape">{}</td>
                            <td><span class="status {}">{}</span></td>
                        </tr>"#,
                            supported,
                            display_name,
                            meta.module_type,
                            input_shapes,
                            output_shapes,
                            status_class,
                            status_text
                        )
                    })
                    .collect();

                format!(
                    r#"<div class="component-section" data-has-unsupported="{}">
                        <div class="component-header">
                            <h3>
                                <span class="chevron">▼</span>
                                {}
                            </h3>
                            <div class="component-stats">
                                <span class="stat">{} layers</span>
                                <span class="stat ok">✓ {}</span>
                                {}
                            </div>
                        </div>
                        <div class="component-content">
                            <table>
                                <thead>
                                    <tr>
                                        <th>Layer</th>
                                        <th>Type</th>
                                        <th>Input Shape</th>
                                        <th>Output Shape</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {}
                                </tbody>
                            </table>
                        </div>
                    </div>"#,
                    unsupported_count > 0,
                    component,
                    sorted_layers.len(),
                    supported_count,
                    if unsupported_count > 0 {
                        format!("<span class=\"stat err\">✗ {}</span>", unsupported_count)
                    } else {
                        String::new()
                    },
                    rows
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"
        <div id="componentList">
            {}
        </div>
        
        <script>
            // State
            let currentFilter = 'all';
            let searchQuery = '';

            // Toggle component sections
            function setupCollapsibles() {{
                document.querySelectorAll('.component-header').forEach(header => {{
                    // Remove old listener if any
                    const newHeader = header.cloneNode(true);
                    header.parentNode.replaceChild(newHeader, header);
                    
                    newHeader.addEventListener('click', () => {{
                        newHeader.classList.toggle('collapsed');
                        newHeader.nextElementSibling.classList.toggle('collapsed');
                    }});
                }});
            }}
            setupCollapsibles();
            
            function updateVisibility() {{
                document.querySelectorAll('.component-section').forEach(section => {{
                    const rows = section.querySelectorAll('.layer-row');
                    let visibleRowsInSection = 0;
                    
                    rows.forEach(row => {{
                        const isSupported = row.dataset.supported === 'true';
                        let matchesFilter = true;
                        
                        if (currentFilter === 'supported') matchesFilter = isSupported;
                        else if (currentFilter === 'unsupported') matchesFilter = !isSupported;
                        
                        const matchesSearch = !searchQuery || row.textContent.toLowerCase().includes(searchQuery) || 
                                           section.querySelector('h3').textContent.toLowerCase().includes(searchQuery);
                        
                        const isVisible = matchesFilter && matchesSearch;
                        row.classList.toggle('hidden', !isVisible);
                        if (isVisible) visibleRowsInSection++;
                    }});
                    
                    section.classList.toggle('hidden', visibleRowsInSection === 0);
                    
                    // Auto-expand if we are filtering for unsupported and there are some
                    if (currentFilter === 'unsupported' && visibleRowsInSection > 0) {{
                        section.querySelector('.component-header').classList.remove('collapsed');
                        section.querySelector('.component-content').classList.remove('collapsed');
                    }}
                }});
            }}

            // Filter buttons
            document.querySelectorAll('.filter-btn').forEach(btn => {{
                btn.addEventListener('click', () => {{
                    const filter = btn.dataset.filter;
                    
                    if (filter === 'expand') {{
                        document.querySelectorAll('.component-header').forEach(h => h.classList.remove('collapsed'));
                        document.querySelectorAll('.component-content').forEach(c => c.classList.remove('collapsed'));
                        return;
                    }}
                    if (filter === 'collapse') {{
                        document.querySelectorAll('.component-header').forEach(h => h.classList.add('collapsed'));
                        document.querySelectorAll('.component-content').forEach(c => c.classList.add('collapsed'));
                        return;
                    }}
                    
                    document.querySelectorAll('.filter-btn').forEach(b => {{
                        if (['all', 'supported', 'unsupported'].includes(b.dataset.filter)) {{
                            b.classList.remove('active');
                        }}
                    }});
                    btn.classList.add('active');
                    
                    currentFilter = filter;
                    updateVisibility();
                }});
            }});
            
            // Search
            document.getElementById('searchBox').addEventListener('input', (e) => {{
                searchQuery = e.target.value.toLowerCase();
                updateVisibility();
            }});
        </script>
        "#,
            components_html
        )
    }
}
