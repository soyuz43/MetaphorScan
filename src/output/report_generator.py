# MetaphorScan/src/output/report_generator.py
"""
PDF report generator using reportlab for MetaphorScan analysis results.
Creates comprehensive reports with highlighted metaphors, attractor basins,
and explanations tied to *Epistemic Autoimmunity*, *Nuremberg Defense*, etc.

Implements transparent reporting (*Epistemic Autoimmunity*, Section 5) to 
expose metaphorical patterns rather than obscuring them.
"""
import os
import logging
import yaml
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

from reportlab.lib.pagesizes import letter, A4  # noqa: F401 (A4 kept for future option)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import Color, black, white, yellow, red, blue, green, grey
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, KeepTogether, Image  # noqa: F401 (KeepTogether/Image kept for future)
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing  # noqa: F401
from reportlab.graphics.charts.barcharts import VerticalBarChart  # noqa: F401
from reportlab.graphics.charts.piecharts import Pie  # noqa: F401

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_settings(config_path="src/config/settings.yaml"):
    """Load report configuration settings."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Settings file not found, using defaults")
        return {
            "output": {
                "report_format": "pdf",
                "highlight_colors": {
                    "sedative": "yellow",
                    "prophylactic": "red"
                }
            }
        }

def get_color_from_name(color_name):
    """Convert color name to reportlab Color object."""
    color_map = {
        'yellow': yellow,
        'red': red,
        'blue': blue,
        'green': green,
        'grey': grey,
        'black': black,
        'white': white
    }
    return color_map.get(color_name.lower(), grey)

class MetaphorScanReportGenerator:
    """
    Comprehensive PDF report generator for MetaphorScan analysis.
    
    Creates structured reports documenting sedative/prophylactic metaphors
    with detailed explanations connecting findings to critical AI literature.
    """
    
    def __init__(self, output_path: str):
        """Initialize report generator with output path."""
        self.output_path = Path(output_path)
        self.settings = load_settings()
        self.highlight_colors = self.settings.get("output", {}).get("highlight_colors", {})
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Document content storage
        self.story: List[Any] = []
        
    def _setup_custom_styles(self):
        """Set up custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=black
        ))
        
        # Section heading style
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=black
        ))
        
        # Subsection heading style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=black
        ))
        
        # Metaphor highlight style for sedative
        self.styles.add(ParagraphStyle(
            name='MetaphorSedative',
            parent=self.styles['Normal'],
            backColor=get_color_from_name(self.highlight_colors.get('sedative', 'yellow')),
            borderColor=black,
            borderWidth=1,
            borderPadding=3
        ))
        
        # Metaphor highlight style for prophylactic
        self.styles.add(ParagraphStyle(
            name='MetaphorProphylactic',
            parent=self.styles['Normal'],
            backColor=get_color_from_name(self.highlight_colors.get('prophylactic', 'red')),
            borderColor=black,
            borderWidth=1,
            borderPadding=3
        ))
        
        # Warning style for attractor basins
        self.styles.add(ParagraphStyle(
            name='AttractorBasinWarning',
            parent=self.styles['Normal'],
            backColor=Color(1, 0.9, 0.9),  # Light red background
            borderColor=red,
            borderWidth=2,
            borderPadding=5,
            spaceBefore=10,
            spaceAfter=10
        ))

        # Table styles for wrapping cells
        self.styles.add(ParagraphStyle(
            name="TableHeader",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=11,
            spaceAfter=0,
            wordWrap="CJK",    # allow breaks inside long tokens
            alignment=TA_LEFT
        ))
        self.styles.add(ParagraphStyle(
            name="TableCell",
            parent=self.styles["Normal"],
            fontSize=8,
            leading=10,
            spaceAfter=0,
            wordWrap="CJK",
            alignment=TA_LEFT
        ))

    # ---- small helpers -----------------------------------------------------

    def _p(self, text: str, style_name: str = "TableCell") -> Paragraph:
        """Safe Paragraph: escape HTML specials, keep basic tags if you add them upstream."""
        if text is None:
            text = ""
        # escape &, <, >, " ; leave simple tags if you rely on them upstream
        text = _xml_escape(str(text), {'"': "&quot;"})
        return Paragraph(text, self.styles[style_name])

    # ---- content builders ---------------------------------------------------

    def add_title_page(self, analysis_results: Dict[str, Any], input_file: str):
        """
        Add title page with analysis overview.
        
        Implements transparent reporting (*Epistemic Autoimmunity*, Section 5)
        by clearly stating methodology and theoretical framework.
        """
        # Main title
        self.story.append(Paragraph("MetaphorScan Analysis Report", self.styles['CustomTitle']))
        self.story.append(Spacer(1, 0.5*inch))
        
        # Subtitle with theoretical grounding
        subtitle = """
        Detection of Sedative and Prophylactic Metaphors in AI Discourse<br/>
        <i>Based on *The Nuremberg Defense of AI*, *AI as an Epistemic Void Generator*, 
        and *The Alignment Problem as Epistemic Autoimmunity*</i>
        """
        self.story.append(Paragraph(subtitle, self.styles['Normal']))
        self.story.append(Spacer(1, 0.3*inch))
        
        # Analysis metadata
        metadata = [
            ["Analyzed File:", input_file],
            ["Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Total Metaphors Found:", str(analysis_results.get('total_metaphors', 0))],
            ["Sedative Metaphors:", str(analysis_results.get('sedative_count', 0))],
            ["Prophylactic Metaphors:", str(analysis_results.get('prophylactic_count', 0))],
            ["Attractor Basins:", str(analysis_results.get('attractor_basins', 0))],
            ["Analysis Method:", "Two-stage pipeline: lexical matching + DistilBERT contextual validation"]
        ]
        
        metadata_table = Table(metadata, colWidths=[2.5*inch, 3*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), Color(0.9, 0.9, 0.9)),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        self.story.append(metadata_table)
        self.story.append(Spacer(1, 0.3*inch))
        
        # Theoretical framework explanation
        framework_text = """
        <b>Theoretical Framework:</b><br/><br/>
        This analysis identifies two types of metaphors that shape AI discourse:<br/><br/>
        
        <b>Sedative Metaphors</b> (*Epistemic Autoimmunity*, Section 2): Terms like "hallucination" 
        and "error" that frame AI failures as natural quirks rather than structural design 
        consequences, thereby sedating critical analysis of AI systems.<br/><br/>
        
        <b>Prophylactic Metaphors</b> (*Nuremberg Defense*, *Epistemic Void Generator*): Terms like 
        "intelligence" and "training" that anthropomorphize AI systems, creating epistemic 
        authority through false analogies to human cognition while obscuring mechanical processes.<br/><br/>
        
        <b>Epistemic Attractor Basins</b> (*Epistemic Autoimmunity*, Section 3): Areas of high 
        metaphor density that create zones of conceptual confusion, trapping discourse in 
        unproductive patterns.
        """
        
        self.story.append(Paragraph(framework_text, self.styles['Normal']))
        self.story.append(PageBreak())
    
    def add_executive_summary(self, analysis_results: Dict[str, Any]):
        """Add executive summary of findings."""
        self.story.append(Paragraph("Executive Summary", self.styles['SectionHeading']))
        
        total_metaphors = analysis_results.get('total_metaphors', 0)
        sedative_count = analysis_results.get('sedative_count', 0)
        prophylactic_count = analysis_results.get('prophylactic_count', 0)
        attractor_basins = analysis_results.get('attractor_basins', 0)
        avg_confidence = analysis_results.get('average_confidence', 0.0)
        
        # Summary statistics
        summary_text = f"""
        This analysis identified <b>{total_metaphors} metaphorical expressions</b> that may influence 
        AI discourse through sedative or prophylactic mechanisms:<br/><br/>
        
        • <b>{sedative_count} sedative metaphors</b> that potentially obscure structural issues 
        by framing AI failures as natural phenomena<br/>
        • <b>{prophylactic_count} prophylactic metaphors</b> that may create false epistemic 
        authority through anthropomorphic analogies<br/>
        • <b>{attractor_basins} epistemic attractor basins</b> representing areas of concentrated 
        metaphorical confusion<br/>
        • Average detection confidence: <b>{avg_confidence:.1%}</b><br/><br/>
        
        <i>These findings suggest {'significant' if total_metaphors > 5 else 'moderate' if total_metaphors > 2 else 'minimal'} 
        metaphorical influence on the analyzed text's epistemic structure.</i>
        """
        
        self.story.append(Paragraph(summary_text, self.styles['Normal']))
        
        # Add warning for high metaphor density
        if total_metaphors > 10 or attractor_basins > 2:
            warning_text = """
            <b>⚠ High Metaphor Density Warning:</b><br/>
            This text shows elevated levels of metaphorical language that may significantly 
            impact reader comprehension of AI systems. Consider reviewing for areas where 
            more precise technical language could improve clarity and reduce epistemic confusion.
            """
            self.story.append(Paragraph(warning_text, self.styles['AttractorBasinWarning']))
        
        self.story.append(Spacer(1, 0.2*inch))
    
    def add_metaphor_analysis(self, validated_matches: List[Dict[str, Any]]):
        """Add detailed analysis of detected metaphors."""
        if not validated_matches:
            self.story.append(Paragraph("Detailed Metaphor Analysis", self.styles['SectionHeading']))
            self.story.append(Paragraph("No metaphors detected in the analyzed text.", self.styles['Normal']))
            return
        
        self.story.append(Paragraph("Detailed Metaphor Analysis", self.styles['SectionHeading']))
        
        # Group metaphors by category
        sedative_metaphors = [m for m in validated_matches if m.get('category') == 'sedative']
        prophylactic_metaphors = [m for m in validated_matches if m.get('category') == 'prophylactic']
        
        # Sedative metaphors section
        if sedative_metaphors:
            self.story.append(Paragraph("Sedative Metaphors", self.styles['SubsectionHeading']))
            self.story.append(Paragraph(
                "These metaphors may obscure structural AI issues by framing failures as natural phenomena:",
                self.styles['Normal']
            ))
            self.story.append(Spacer(1, 0.1*inch))
            
            self._add_metaphor_table(sedative_metaphors, 'sedative')
            self.story.append(Spacer(1, 0.2*inch))
        
        # Prophylactic metaphors section  
        if prophylactic_metaphors:
            self.story.append(Paragraph("Prophylactic Metaphors", self.styles['SubsectionHeading']))
            self.story.append(Paragraph(
                "These metaphors may create false epistemic authority through anthropomorphic analogies:",
                self.styles['Normal']
            ))
            self.story.append(Spacer(1, 0.1*inch))
            
            self._add_metaphor_table(prophylactic_metaphors, 'prophylactic')
            self.story.append(Spacer(1, 0.2*inch))
    
    def _add_metaphor_table(self, metaphors: List[Dict[str, Any]], category: str):
        """Add a table of metaphors for a specific category with wrapped cells and page-fit widths."""
        # Headers as Paragraphs
        headers = [
            self._p("Term", "TableHeader"),
            self._p("Replacement", "TableHeader"),
            self._p("Confidence", "TableHeader"),
            self._p("Context", "TableHeader"),
            self._p("Explanation", "TableHeader"),
        ]
        rows: List[List[Any]] = [headers]

        # Data rows as Paragraphs (wrap-enabled)
        for m in metaphors:
            term = m.get('term', '') or ''
            replacement = m.get('replacement', '') or ''
            confidence = f"{m.get('confidence', 0):.1%}"

            ctx = m.get('sentence_context', '') or ''
            if len(ctx) > 220:  # soft cap for readability
                ctx = ctx[:220] + "…"

            desc = m.get('description', '') or ''
            if len(desc) > 300:  # optional: cap super-long explanations
                desc = desc[:300] + "…"

            rows.append([
                self._p(term),
                self._p(replacement),
                self._p(confidence),
                self._p(ctx),
                self._p(desc),
            ])

        # Fit columns to available frame width
        W = getattr(self, "frame_width", 7.0 * inch)  # fallback if not set
        weights = [0.12, 0.18, 0.10, 0.30, 0.30]      # Term, Replacement, Confidence, Context, Explanation
        col_widths = [W * w for w in weights]

        table = Table(
            rows,
            colWidths=col_widths,
            repeatRows=1,   # header repeats on page breaks
            splitByRow=1    # allow splitting between rows
        )
        table.hAlign = "LEFT"

        # Style table
        table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), Color(0.85, 0.85, 0.85)),
            ('TEXTCOLOR', (0, 0), (-1, 0), black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, Color(0.96, 0.96, 0.96)]),
        ]

        # Highlight term column with category color
        if category == 'sedative':
            table_style.append(('BACKGROUND', (0, 1), (0, -1),
                                get_color_from_name(self.highlight_colors.get('sedative', 'yellow'))))
        elif category == 'prophylactic':
            table_style.append(('BACKGROUND', (0, 1), (0, -1),
                                get_color_from_name(self.highlight_colors.get('prophylactic', 'red'))))

        table.setStyle(TableStyle(table_style))
        self.story.append(table)

    def add_attractor_basin_analysis(self, attractor_basins: List[Dict[str, Any]], semantic_basins: List[Dict[str, Any]] = None):
        """Add analysis of epistemic attractor basins."""
        all_basins: List[Dict[str, Any]] = []
        if attractor_basins:
            all_basins.extend(attractor_basins)
        if semantic_basins:
            all_basins.extend(semantic_basins)
        
        if not all_basins:
            return
        
        self.story.append(Paragraph("Epistemic Attractor Basin Analysis", self.styles['SectionHeading']))
        
        basin_intro = """
        <b>Epistemic Attractor Basins</b> (*Epistemic Autoimmunity*, Section 3) are areas where 
        multiple metaphors cluster together, creating zones of conceptual confusion that can 
        trap discourse in unproductive patterns. High-density metaphor clusters may indicate 
        areas where more precise technical language would improve clarity.<br/><br/>
        
        The following basins were detected:
        """
        self.story.append(Paragraph(basin_intro, self.styles['Normal']))
        self.story.append(Spacer(1, 0.1*inch))
        
        for i, basin in enumerate(all_basins, 1):
            basin_text = f"""
            <b>Basin {i}:</b> {basin.get('metaphor_count', 0)} metaphors 
            {'(Semantic coherence: ' + f"{basin.get('semantic_coherence', 0):.2f}" + ')' if 'semantic_coherence' in basin else ''}<br/>
            <i>Context:</i> { _xml_escape(str(basin.get('sentence', basin.get('description', '')))[:200]) }...<br/>
            <i>Categories:</i> {', '.join(basin.get('categories', []))}<br/>
            <i>Metaphors:</i> {', '.join([m.get('term', '') for m in basin.get('metaphors', [])])}<br/>
            """
            
            self.story.append(Paragraph(basin_text, self.styles['AttractorBasinWarning']))
            self.story.append(Spacer(1, 0.1*inch))
    
    def add_highlighted_text(self, original_text: str, validated_matches: List[Dict[str, Any]]):
        """Add section with highlighted original text showing detected metaphors."""
        if not validated_matches:
            return
            
        self.story.append(Paragraph("Highlighted Text Analysis", self.styles['SectionHeading']))
        self.story.append(Paragraph(
            "The following shows the original text with detected metaphors highlighted:",
            self.styles['Normal']
        ))
        self.story.append(Spacer(1, 0.1*inch))
        
        # Create highlighted version of text
        highlighted_text = self._create_highlighted_text(original_text, validated_matches)
        
        # Split into chunks to avoid paragraph size limits
        text_chunks = self._split_text_for_display(highlighted_text, max_length=2000)
        
        for chunk in text_chunks:
            self.story.append(Paragraph(chunk, self.styles['Normal']))
            self.story.append(Spacer(1, 0.1*inch))
    
    def _create_highlighted_text(self, text: str, matches: List[Dict[str, Any]]) -> str:
        """Create HTML-highlighted version of text with metaphor annotations."""
        highlighted_text = _xml_escape(text) + "\n\n<b>Detected Metaphors:</b>\n"
        
        # Group matches by unique terms
        term_counts: Dict[str, Dict[str, Any]] = {}
        for match in matches:
            term = match.get('term', '')
            category = match.get('category', '')
            replacement = match.get('replacement', '')
            
            key = f"{term}_{category}"
            if key not in term_counts:
                term_counts[key] = {
                    'term': term,
                    'category': category,
                    'replacement': replacement,
                    'count': 0
                }
            term_counts[key]['count'] += 1
        
        # Add metaphor summary
        for item in term_counts.values():
            term = _xml_escape(item['term'])
            category = item['category']
            replacement = _xml_escape(item['replacement'])
            count = item['count']
            
            if category == 'sedative':
                color_desc = "SEDATIVE (yellow highlight)"
            elif category == 'prophylactic':
                color_desc = "PROPHYLACTIC (red highlight)"
            else:
                color_desc = "UNKNOWN"
            
            count_text = f" (appears {count} times)" if count > 1 else ""
            highlighted_text += f"\n• '<b>{term}</b>' → '<i>{replacement}</i>' [{color_desc}]{count_text}"
        
        return highlighted_text
    
    def _split_text_for_display(self, text: str, max_length: int = 2000) -> List[str]:
        """Split long text into manageable chunks for PDF display."""
        if len(text) <= max_length:
            return [text]
        
        chunks: List[str] = []
        current_pos = 0
        
        while current_pos < len(text):
            end_pos = min(current_pos + max_length, len(text))
            
            # Try to break at sentence boundary
            if end_pos < len(text):
                # Look for sentence ending near the break point
                for i in range(end_pos, max(current_pos, end_pos - 200), -1):
                    if text[i - 1] in '.!?':
                        end_pos = i
                        break
            
            chunk = text[current_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)
            
            current_pos = end_pos
        
        return chunks
    
    def add_methodology_section(self):
        """Add detailed methodology explanation."""
        self.story.append(Paragraph("Analysis Methodology", self.styles['SectionHeading']))
        
        methodology_text = """
        <b>Two-Stage Detection Pipeline:</b><br/><br/>
        
        <b>Stage 1: Lexical Matching</b><br/>
        • Uses spaCy for dependency parsing and context-aware term matching<br/>
        • Scores metaphors based on proximity to AI-related technical terms<br/>
        • Implements rule-based detection from predefined lexicon<br/>
        • Calculates confidence based on linguistic features and context<br/><br/>
        
        <b>Stage 2: Contextual Validation</b><br/>
        • Uses DistilBERT embeddings for semantic similarity analysis<br/>
        • Validates AI domain context without opaque classification<br/>
        • Checks metaphor-category semantic coherence<br/>
        • Detects semantic attractor basins through clustering<br/><br/>
        
        <b>Theoretical Grounding:</b><br/>
        • <i>Sedative metaphors</i>: Terms that obscure structural issues (*Epistemic Autoimmunity*)<br/>
        • <i>Prophylactic metaphors</i>: Terms that create false epistemic authority (*Nuremberg Defense*)<br/>
        • <i>Attractor basins</i>: High-density metaphor clusters creating confusion zones<br/><br/>
        
        <b>Transparency Principle:</b><br/>
        All analysis steps use interpretable methods avoiding black-box AI classification, 
        following the critique of epistemic opacity in *The Alignment Problem as Epistemic Autoimmunity*.
        """
        
        self.story.append(Paragraph(methodology_text, self.styles['Normal']))
        self.story.append(PageBreak())
    
    def add_recommendations(self, analysis_results: Dict[str, Any]):
        """Add recommendations based on analysis results."""
        self.story.append(Paragraph("Recommendations", self.styles['SectionHeading']))
        
        total_metaphors = analysis_results.get('total_metaphors', 0)
        sedative_count = analysis_results.get('sedative_count', 0)
        prophylactic_count = analysis_results.get('prophylactic_count', 0)
        attractor_basins = analysis_results.get('attractor_basins', 0)
        
        recommendations: List[str] = []
        
        if sedative_count > 0:
            recommendations.append(
                f"<b>Sedative Metaphor Awareness:</b> {sedative_count} sedative metaphors were detected. "
                "Consider whether these terms might be obscuring important structural discussions about AI system design and failure modes."
            )
        
        if prophylactic_count > 0:
            recommendations.append(
                f"<b>Prophylactic Metaphor Review:</b> {prophylactic_count} prophylactic metaphors were found. "
                "Evaluate whether anthropomorphic language might be creating false impressions of AI capabilities or consciousness."
            )
        
        if attractor_basins > 0:
            recommendations.append(
                f"<b>Attractor Basin Attention:</b> {attractor_basins} epistemic attractor basins were identified. "
                "These areas of high metaphor density may benefit from more precise technical language."
            )
        
        if total_metaphors > 10:
            recommendations.append(
                "<b>High Metaphor Density:</b> This text shows elevated metaphorical language usage. "
                "Consider reviewing for opportunities to use more direct technical terminology where precision is important."
            )
        
        if not recommendations:
            recommendations.append(
                "<b>Low Metaphor Usage:</b> This text shows minimal metaphorical language that might influence AI discourse understanding."
            )
        
        recommendations.append(
            "<b>General Recommendation:</b> Use this analysis as one input among many when evaluating "
            "the clarity and precision of AI-related discourse. Metaphors can be valuable for communication "
            "but may sometimes obscure important technical or ethical considerations."
        )
        
        for rec in recommendations:
            self.story.append(Paragraph(f"• {rec}", self.styles['Normal']))
            self.story.append(Spacer(1, 0.1*inch))
    
    def generate_report(self, analysis_results: Dict[str, Any], input_file: str,
                        original_text: str = "") -> str:
        """
        Generate complete PDF report from analysis results.

        Args:
            analysis_results: Combined results from pipeline analysis
            input_file: Path to original input file
            original_text: Original text content for highlighting

        Returns:
            str: Path to generated PDF report
        """
        logger.info(f"Generating PDF report: {self.output_path}")

        # Extract data from analysis results
        if 'validated_matches' in analysis_results:
            # New format from contextual analyzer
            validated_matches = analysis_results['validated_matches']
            attractor_basins = analysis_results.get('attractor_basins', [])
            semantic_basins = analysis_results.get('semantic_basins', [])
            stats = analysis_results.get('statistics', {})
        else:
            # Legacy format - assume these are validated matches
            validated_matches = analysis_results if isinstance(analysis_results, list) else []
            attractor_basins = []
            semantic_basins = []
            stats = {}

        # Compile summary statistics
        summary_stats = {
            'total_metaphors': len(validated_matches),
            'sedative_count': len([m for m in validated_matches if m.get('category') == 'sedative']),
            'prophylactic_count': len([m for m in validated_matches if m.get('category') == 'prophylactic']),
            'attractor_basins': len(attractor_basins) + len(semantic_basins),
            'average_confidence': (
                sum(m.get('confidence', 0) for m in validated_matches) /
                max(len(validated_matches), 1)
            ),
        }
        summary_stats.update(stats)  # Add any additional stats

        # Create PDF document (slightly tighter margins to fit wrapped text)
        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=letter,
            rightMargin=0.6 * inch,
            leftMargin=0.6 * inch,
            topMargin=0.9 * inch,
            bottomMargin=0.6 * inch,
        )

        # Provide frame width to table builder so it fits the page
        self.frame_width = doc.width

        # Build report content
        self.add_title_page(summary_stats, input_file)
        self.add_executive_summary(summary_stats)
        self.add_metaphor_analysis(validated_matches)
        self.add_attractor_basin_analysis(attractor_basins, semantic_basins)

        if original_text:
            self.add_highlighted_text(original_text, validated_matches)

        self.add_methodology_section()
        self.add_recommendations(summary_stats)

        # Generate PDF
        try:
            doc.build(self.story)
            logger.info(f"PDF report generated successfully: {self.output_path}")
            return str(self.output_path)
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            raise


# Convenience function for direct usage
def generate_metaphor_report(analysis_results: Dict[str, Any], output_path: str, 
                             input_file: str, original_text: str = "") -> str:
    """
    Generate a MetaphorScan analysis report.
    
    Args:
        analysis_results: Results from MetaphorScan pipeline
        output_path: Path for output PDF file
        input_file: Original input file path
        original_text: Original text content
        
    Returns:
        str: Path to generated PDF report
        
    Implements transparent reporting (*Epistemic Autoimmunity*, Section 5) of
    metaphorical patterns in AI discourse.
    """
    generator = MetaphorScanReportGenerator(output_path)
    return generator.generate_report(analysis_results, input_file, original_text)

# Test function for development
if __name__ == "__main__":
    # Test report generation with sample data
    sample_results = {
        'validated_matches': [
            {
                'term': 'hallucination',
                'category': 'sedative',
                'replacement': 'fabrication',
                'confidence': 0.85,
                'description': 'Implies truth-seeking failure, obscuring design consequence (*Epistemic Autoimmunity*, Section 2).',
                'sentence_context': 'The AI model hallucinated incorrect outputs during inference.',
                'position': {'start': 12, 'end': 25}
            },
            {
                'term': 'intelligence',
                'category': 'prophylactic',
                'replacement': 'pattern simulation',
                'confidence': 0.78,
                'description': 'Implies sentience, misdirecting discourse (*Epistemic Void Generator*).',
                'sentence_context': 'The system showed remarkable artificial intelligence capabilities.',
                'position': {'start': 35, 'end': 47}
            }
        ],
        'attractor_basins': [
            {
                'sentence': 'The intelligent model hallucinated during training.',
                'metaphor_count': 2,
                'categories': ['sedative', 'prophylactic'],
                'metaphors': []
            }
        ],
        'statistics': {
            'total_metaphors': 2,
            'sedative_count': 1,
            'prophylactic_count': 1,
            'average_confidence': 0.815
        }
    }
    
    sample_text = """
    The artificial intelligence model demonstrated remarkable learning capabilities during training.
    However, the system occasionally hallucinated outputs that diverged from expected patterns.
    """
    
    try:
        # Ensure output directory exists
        os.makedirs("outputs/reports", exist_ok=True)
        
        # Generate test report
        report_path = generate_metaphor_report(
            sample_results,
            "outputs/reports/test_report.pdf",
            "sample_text.txt",
            sample_text
        )
        print(f"Test report generated: {report_path}")
        
    except Exception as e:
        print(f"Error generating test report: {e}")
        import traceback
        traceback.print_exc()
