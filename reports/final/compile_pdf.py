import os
from markdown_pdf import MarkdownPdf, Section

def main():
    # Resolve paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_md = os.path.join(base_dir, 'compiled_variance_risk_premium_report.md')
    output_pdf = os.path.join(base_dir, 'compiled_variance_risk_premium_report.pdf')
    
    print(f"Reading markdown from: {input_md}")
    with open(input_md, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Resolve relative image paths to absolute paths
    # The markdown contains paths like `../figures/...`
    figures_dir = os.path.abspath(os.path.join(base_dir, '..', 'figures')).replace('\\', '/')
    
    # Replace all instances of `../figures/` with the absolute path
    md_content = md_content.replace('../figures/', f'{figures_dir}/')
    
    print("Converting Markdown to PDF...")
    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(md_content, toc=False))
    
    pdf.save(output_pdf)
    print(f"Successfully compiled PDF to: {output_pdf}")

if __name__ == "__main__":
    main()
