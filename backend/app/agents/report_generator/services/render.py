import pdfkit
import shutil
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape

def render_html_report(
    title: str,
    period: str,
    summary: str,
    chart_rel_path: str | None,
    sources: list[dict],
    disclaimer: str,
    reports_dir: str,
    templates_dir: str,
    filename_slug: str,
) -> str:
    """
    Renders templates/report.html to artifacts/reports/<slug>.html
    Returns a RELATIVE path beginning with /agents/...
    """
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape()
    )
    tmpl = env.get_template("report.html")

    html = tmpl.render(
        title=title,
        period=period,
        summary=summary,
        chart_path=chart_rel_path,
        sources=sources,
        disclaimer=disclaimer,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )

    out_file = Path(reports_dir) / f"{filename_slug}.html"
    out_file.write_text(html, encoding="utf-8")

    # relative URL for UI/use in response
    return f"/agents/report_generator/artifacts/reports/{filename_slug}.html"


def html_to_pdf(html_path: str, pdf_out_path: str, wkhtmltopdf_path: str | None = None, css_path: str | None = None) -> bool:
    """
    Convert a local HTML file to PDF. Returns True on success.
    css_path: absolute path to the CSS to apply (recommended for consistent styling)
    """
    exe = wkhtmltopdf_path or shutil.which("wkhtmltopdf")
    if not exe:
        return False

    options = {
        "quiet": "",
        "enable-local-file-access": "",   # allow local images/css
        "page-size": "A4",
        "margin-top": "12mm",
        "margin-right": "12mm",
        "margin-bottom": "14mm",
        "margin-left": "12mm",
        "print-media-type": "",           # use @media print rules
        "encoding": "UTF-8",
        "load-error-handling": "ignore",
    }
    config = pdfkit.configuration(wkhtmltopdf=exe)

    if css_path:
        pdfkit.from_file(html_path, pdf_out_path, options=options, configuration=config, css=css_path)
    else:
        pdfkit.from_file(html_path, pdf_out_path, options=options, configuration=config)
    return True

def html_to_pdf_strict(
    html_path: str,
    pdf_out_path: str,
    css_abs_path: str,
    chart_abs_path: str | None,
    wkhtmltopdf_path: str | None = None,
) -> bool:
    """
    Convert the already-rendered HTML to a PDF by:
    - inlining the CSS (so styling always applies)
    - replacing the chart <img> relative src with an absolute file:// path
    Returns True on success.
    """
    exe = wkhtmltopdf_path or shutil.which("wkhtmltopdf")
    if not exe:
        return False

    # 1) Read HTML + CSS
    html = Path(html_path).read_text(encoding="utf-8")
    css  = Path(css_abs_path).read_text(encoding="utf-8")

    # 2) Inline CSS: replace the <link rel="stylesheet" ...> with <style>...</style>
    html = html.replace(
        '<link rel="stylesheet" href="../../static/css/style.css" />',
        f"<style>\n{css}\n</style>"
    )

    # 3) If we have a chart, swap the relative src with an absolute file URL
    if chart_abs_path:
        chart_fname = Path(chart_abs_path).name
        html = html.replace(
            f'src="../charts/{chart_fname}"',
            f'src="file:///{Path(chart_abs_path).as_posix()}"'
        )

    # 4) wkhtmltopdf options
    options = {
        "quiet": "",
        "enable-local-file-access": "",   # allow local files (img/css)
        "page-size": "A4",
        "margin-top": "12mm",
        "margin-right": "12mm",
        "margin-bottom": "14mm",
        "margin-left": "12mm",
        "print-media-type": "",
        "encoding": "UTF-8",
        "load-error-handling": "ignore",
    }
    config = pdfkit.configuration(wkhtmltopdf=exe)

    # 5) Render from the in-memory HTML string (so inline CSS applies)
    pdfkit.from_string(html, pdf_out_path, options=options, configuration=config)
    return True
