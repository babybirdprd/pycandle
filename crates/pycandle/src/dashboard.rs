use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use pycandle_core::ComparisonResult;
use ratatui::{
    prelude::*,
    widgets::{
        Block, Borders, List, ListItem, Paragraph,
        canvas::{Canvas, Rectangle},
    },
};
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

pub enum DashboardEvent {
    Input(event::KeyEvent),
    TestLine(String),
    Tick,
}

struct App {
    test_results: Vec<TestResult>,
    parity_results: HashMap<String, ComparisonResult>,
    output_log: Vec<String>,
    running: bool,
    scroll: usize,
    passed: usize,
    failed: usize,
    total: usize,
}

struct TestResult {
    name: String,
    status: TestStatus,
    details: String,
}

enum TestStatus {
    Running,
    Passed,
    Failed,
}

impl App {
    fn new() -> Self {
        Self {
            test_results: Vec::new(),
            parity_results: HashMap::new(),
            output_log: Vec::new(),
            running: true,
            scroll: 0,
            passed: 0,
            failed: 0,
            total: 0,
        }
    }

    fn on_tick(&mut self) {
        // Periodically reload parity results
        self.load_parity_results();
    }

    fn load_parity_results(&mut self) {
        if let Ok(file) = std::fs::File::open("verification_results.jsonl") {
            let reader = BufReader::new(file);
            for line in reader.lines() {
                if let Ok(l) = line {
                    if let Ok(res) = serde_json::from_str::<ComparisonResult>(&l) {
                        // Normalize name to help matching?
                        // For now just store by layer name
                        self.parity_results.insert(res.name.clone(), res);
                    }
                }
            }
        }
    }

    fn on_log(&mut self, line: String) {
        // Parse cargo test output
        if line.contains("test result: ok") {
            // Summary line, maybe parse stats?
        } else if line.contains("test ") && line.contains(" ... ok") {
            let name = line.replace("test ", "").replace(" ... ok", "");
            self.test_results.push(TestResult {
                name,
                status: TestStatus::Passed,
                details: "".to_string(),
            });
            self.passed += 1;
            self.total += 1;
        } else if line.contains("test ") && line.contains(" ... FAILED") {
            let name = line.replace("test ", "").replace(" ... FAILED", "");
            self.test_results.push(TestResult {
                name,
                status: TestStatus::Failed,
                details: "".to_string(),
            });
            self.failed += 1;
            self.total += 1;
        }

        // Keep a rolling log
        self.output_log.push(line);
        if self.output_log.len() > 100 {
            self.output_log.remove(0);
        }
    }
}

pub fn run_dashboard(_args: &[String]) -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Channels
    let (tx, rx) = mpsc::channel();
    let tick_rate = Duration::from_millis(100);

    // Spawn Input Thread
    let tx_input = tx.clone();
    thread::spawn(move || {
        loop {
            if event::poll(Duration::from_millis(50)).unwrap() {
                if let Event::Key(key) = event::read().unwrap() {
                    if key.kind == KeyEventKind::Press {
                        if tx_input.send(DashboardEvent::Input(key)).is_err() {
                            return;
                        }
                    }
                }
            }
        }
    });

    // Spawn Tick Thread
    let tx_tick = tx.clone();
    thread::spawn(move || {
        loop {
            thread::sleep(tick_rate);
            if tx_tick.send(DashboardEvent::Tick).is_err() {
                return;
            }
        }
    });

    // Spawn Test Runner Thread
    let tx_log = tx.clone();
    thread::spawn(move || {
        let mut cmd = Command::new("cargo")
            .arg("test")
            .arg("--")
            .arg("--nocapture") // Important to see output in real-time
            .stdout(Stdio::piped())
            .stderr(Stdio::piped()) // Capture stderr too
            .spawn()
            .expect("Failed to spawn cargo test");

        if let Some(stdout) = cmd.stdout.take() {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if let Ok(l) = line {
                    let _ = tx_log.send(DashboardEvent::TestLine(l));
                }
            }
        }
    });

    // App Loop
    let mut app = App::new();
    let mut should_quit = false;

    while !should_quit {
        terminal.draw(|f| ui(f, &mut app))?;

        match rx.recv()? {
            DashboardEvent::Input(key) => match key.code {
                KeyCode::Char('q') => should_quit = true,
                KeyCode::Down => {
                    if app.scroll < app.test_results.len().saturating_sub(1) {
                        app.scroll += 1;
                    }
                }
                KeyCode::Up => {
                    if app.scroll > 0 {
                        app.scroll -= 1;
                    }
                }
                _ => {}
            },
            DashboardEvent::TestLine(line) => app.on_log(line),
            DashboardEvent::Tick => app.on_tick(),
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}

fn ui(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(0),    // Main (List + Details)
            Constraint::Length(8), // Log
            Constraint::Length(1), // Footer
        ])
        .split(f.area());

    // Header
    let title = Paragraph::new(format!(
        "Parity Dashboard | Total: {} | Passed: {} | Failed: {}",
        app.total, app.passed, app.failed
    ))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("PyCandle Status"),
    )
    .style(Style::default().fg(Color::Cyan));
    f.render_widget(title, chunks[0]);

    // Main Content Split
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(chunks[1]);

    // --- List Widget (Left) ---
    let items: Vec<ListItem> = app
        .test_results
        .iter()
        .enumerate()
        .skip(app.scroll.saturating_sub(10)) // Simple pseudo-windowing
        .take(50)
        .map(|(i, res)| {
            let style = match res.status {
                TestStatus::Passed => Style::default().fg(Color::Green),
                TestStatus::Failed => Style::default().fg(Color::Red),
                TestStatus::Running => Style::default().fg(Color::Yellow),
            };
            let icon = match res.status {
                TestStatus::Passed => "✔",
                TestStatus::Failed => "✖",
                TestStatus::Running => "⏳",
            };

            let prefix = if i == app.scroll { "> " } else { "  " };
            ListItem::new(format!("{}{}{}", prefix, icon, res.name)).style(if i == app.scroll {
                style.add_modifier(Modifier::BOLD | Modifier::REVERSED)
            } else {
                style
            })
        })
        .collect();

    let list_title = if app.test_results.is_empty() {
        "Running Tests..."
    } else {
        "Tests"
    };
    let list = List::new(items).block(Block::default().borders(Borders::ALL).title(list_title));
    f.render_widget(list, main_chunks[0]);

    // --- Details Widget (Right) ---
    // Try to find matching parity result for the selected test
    let selected_test = app.test_results.get(app.scroll);

    let details_block = Block::default().borders(Borders::ALL).title("Details");
    let inner = details_block.inner(main_chunks[1]);
    f.render_widget(details_block, main_chunks[1]);

    let details_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(5), Constraint::Min(0)])
        .split(inner);

    if let Some(test) = selected_test {
        // Find result by fuzzy matching name
        let matched_result = app.parity_results.values().find(|r| {
            let parts: Vec<&str> = r.name.split('.').collect();
            if parts.len() >= 2 {
                test.name.contains(parts[parts.len() - 1])
                    && test.name.contains(parts[parts.len() - 2])
            } else {
                test.name.contains(&r.name)
            }
        });

        if let Some(res) = matched_result {
            // Stats
            let status_line = Line::from(vec![
                Span::raw(format!("Layer: {}\nStatus: ", res.name)),
                if res.passed {
                    Span::styled("PASS", Style::default().fg(Color::Green))
                } else {
                    Span::styled("FAIL", Style::default().fg(Color::Red))
                },
                Span::raw(format!(
                    "\nMSE: {:.2e}\nCosSim: {:.6}",
                    res.mse, res.cosine_sim
                )),
            ]);

            f.render_widget(Paragraph::new(status_line), details_layout[0]);

            // Heatmap
            if let Some(heatmap) = &res.heatmap {
                // Render 8x8 heatmap
                if heatmap.len() == 64 {
                    // Normalize for coloring
                    let max_val = heatmap.iter().cloned().fold(0.0, f32::max);

                    let canvas = Canvas::default()
                        .block(
                            Block::default()
                                .borders(Borders::ALL)
                                .title("Error Heatmap (8x8)"),
                        )
                        .x_bounds([0.0, 8.0])
                        .y_bounds([0.0, 8.0])
                        .paint(move |ctx| {
                            for r in 0..8 {
                                for c in 0..8 {
                                    let idx = r * 8 + c;
                                    let val = heatmap[idx];
                                    let intensity = if max_val > 0.0 { val / max_val } else { 0.0 };

                                    // Color ramp: Black -> Blue -> Red
                                    let color = if intensity < 0.2 {
                                        Color::DarkGray
                                    } else if intensity < 0.5 {
                                        Color::Blue
                                    } else if intensity < 0.8 {
                                        Color::Magenta
                                    } else {
                                        Color::Red
                                    };

                                    // Draw 1x1 rectangle
                                    // r=0 is first row (top), so map to y=7
                                    ctx.draw(&Rectangle {
                                        x: c as f64,
                                        y: 7.0 - r as f64,
                                        width: 1.0,
                                        height: 1.0,
                                        color,
                                    });
                                }
                            }
                        });
                    f.render_widget(canvas, details_layout[1]);
                }
            } else if !res.passed {
                f.render_widget(Paragraph::new("No heatmap available."), details_layout[1]);
            }
        } else {
            f.render_widget(
                Paragraph::new("No trace data found for this test."),
                details_layout[0],
            );
        }
    }

    // Footer
    let footer = Paragraph::new("Press 'q' to quit | '↑/↓' to scroll")
        .style(Style::default().fg(Color::Gray));
    f.render_widget(footer, chunks[3]);

    // Log output (Raw)
    let log_items: Vec<ListItem> = app
        .output_log
        .iter()
        .rev()
        .take(8)
        .rev()
        .map(|l| ListItem::new(l.clone()).style(Style::default().fg(Color::DarkGray)))
        .collect();

    let log_box =
        List::new(log_items).block(Block::default().borders(Borders::ALL).title("Raw Output"));
    f.render_widget(log_box, chunks[2]);
}
