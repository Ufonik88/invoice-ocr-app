"""
Invoice OCR Desktop Application - Main Streamlit UI

A full-stack desktop application for extracting invoice data using OCR
and LLMs, with export to Excel.

Features:
- File upload (PDF, images)
- OCR with preprocessing options
- LLM extraction (Ollama, LM Studio, Deepseek)
- Vision model support (llava)
- Editable data review
- Excel export with ZAR formatting
"""

import logging
import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import (
    AppConfig,
    LLMProvider,
    get_config,
    validate_system_requirements,
)
from app.export.excel import ExcelExporter
from app.llm.client import LLMClient, LLMClientError
from app.llm.parser import InvoiceParser
from app.models.invoice import ExtractedInvoice, ExtractionResult, LineItem
from app.ocr.extractor import OCRExtractor
from app.ocr.preprocessor import PreprocessingLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def apply_theme():
    """Apply a modern, accessible visual theme."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=Atkinson+Hyperlegible:wght@400;700&display=swap');

            :root {
                --primary: #0C6E6B;
                --primary-strong: #0A5A57;
                --accent: #F06543;
                --bg: #F5EFE6;
                --surface: #FFFFFF;
                --text: #1D1F21;
                --muted: #5B5E62;
                --border: #E6E0D7;
                --success: #1B9C5A;
                --warning: #C47F00;
                --error: #B00020;
            }

            html, body, [class*="css"] {
                font-family: 'Atkinson Hyperlegible', 'Source Sans 3', system-ui, -apple-system, sans-serif;
                color: var(--text);
            }

            .stApp {
                background: linear-gradient(180deg, #F9F5EF 0%, #F5EFE6 45%, #F2E9DD 100%);
            }

            .app-header {
                background: var(--surface);
                border: 1px solid var(--border);
                padding: 1.25rem 1.5rem;
                border-radius: 16px;
                box-shadow: 0 2px 12px rgba(20, 20, 20, 0.08);
                margin-bottom: 1rem;
            }

            .app-card {
                background: var(--surface);
                border: 1px solid var(--border);
                padding: 1rem 1.25rem;
                border-radius: 14px;
            }

            .muted {
                color: var(--muted);
            }

            .step-row {
                display: flex;
                gap: 0.5rem;
                flex-wrap: wrap;
                margin-bottom: 1rem;
            }

            .step-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background: #EAF3F2;
                color: #0B4E4B;
                border: 1px solid #CDE3E1;
                padding: 0.25rem 0.6rem;
                border-radius: 999px;
                font-size: 0.85rem;
                font-weight: 600;
            }

            .step-pill.inactive {
                background: #F1F1F1;
                color: #6B6E72;
                border-color: #E3E3E3;
            }

            .primary-cta button {
                background: var(--primary) !important;
                color: #FFFFFF !important;
                border: none !important;
            }

            .primary-cta button:hover {
                background: var(--primary-strong) !important;
            }

            .secondary-cta button {
                border-color: var(--primary) !important;
                color: var(--primary) !important;
            }

            .help-callout {
                background: #FFF7ED;
                border: 1px solid #FCD6B1;
                color: #7C4A00;
                padding: 0.75rem 1rem;
                border-radius: 12px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state():
    """Initialize Streamlit session state."""
    if "config" not in st.session_state:
        st.session_state.config = get_config()
    
    if "extracted_invoice" not in st.session_state:
        st.session_state.extracted_invoice = None
    
    if "ocr_text" not in st.session_state:
        st.session_state.ocr_text = None
    
    if "uploaded_file_content" not in st.session_state:
        st.session_state.uploaded_file_content = None
    
    if "system_validated" not in st.session_state:
        st.session_state.system_validated = False
    
    if "validation_results" not in st.session_state:
        st.session_state.validation_results = None

    if "export_result_path" not in st.session_state:
        st.session_state.export_result_path = None

    if "last_provider_used" not in st.session_state:
        st.session_state.last_provider_used = None


def validate_system():
    """Validate system requirements on startup."""
    if not st.session_state.system_validated:
        with st.spinner("Validating system requirements..."):
            results = validate_system_requirements()
            st.session_state.validation_results = results
            st.session_state.system_validated = True
    
    return st.session_state.validation_results


def render_system_status_panel():
    """Display system validation status with plain language."""
    results = st.session_state.validation_results
    if not results:
        return

    st.subheader("System check")

    critical_issues = []
    warnings = []

    if not results["tesseract"]["installed"]:
        critical_issues.append(
            "Text scanning tool is missing. Install Tesseract to read invoices."
        )

    if not results["ollama"]["available"] and not results["deepseek"]["configured"]:
        warnings.append(
            "No AI engine is ready. Start Ollama or add a Deepseek API key."
        )

    if critical_issues:
        for issue in critical_issues:
            st.error(issue)

    if warnings:
        for warning in warnings:
            st.warning(warning)

    available = []
    if results["ollama"]["available"]:
        models = results["ollama"].get("models", [])
        available.append(f"Ollama ({len(models)} models)")
    if results["lm_studio"]["available"]:
        available.append("LM Studio")
    if results["deepseek"]["configured"]:
        available.append("Deepseek (cloud)")

    if available:
        st.success(f"Available engines: {', '.join(available)}")


def render_expert_settings_panel():
    """Render expert settings in a collapsible panel."""
    results = st.session_state.validation_results or {}

    st.subheader("Expert mode")
    st.caption("Advanced settings are optional. Most users can keep the defaults.")

    with st.expander("Advanced settings", expanded=False):
        provider_names = {
            LLMProvider.OLLAMA: "Ollama (local)",
            LLMProvider.LM_STUDIO: "LM Studio (local)",
            LLMProvider.DEEPSEEK: "Deepseek (cloud)",
        }

        all_providers = [LLMProvider.OLLAMA, LLMProvider.LM_STUDIO, LLMProvider.DEEPSEEK]

        selected_provider = st.selectbox(
            "AI engine",
            options=all_providers,
            format_func=lambda x: provider_names.get(x, x.value),
            key="provider_selection",
            help="Choose which AI service to use for extraction",
        )
        st.session_state.selected_provider = selected_provider

        if selected_provider == LLMProvider.OLLAMA:
            if results.get("ollama", {}).get("available"):
                st.success("Ollama is running")
            else:
                st.warning("Ollama is not running. Start it with: ollama serve")
        elif selected_provider == LLMProvider.LM_STUDIO:
            if results.get("lm_studio", {}).get("available"):
                st.success("LM Studio is running")
            else:
                st.warning("LM Studio is not running. Start the local server.")
        elif selected_provider == LLMProvider.DEEPSEEK:
            if results.get("deepseek", {}).get("configured"):
                st.success("Deepseek API key is set")
            else:
                st.warning("Deepseek API key is not set in .env")

        st.divider()

        if selected_provider == LLMProvider.OLLAMA:
            st.subheader("Ollama settings")

            models = results.get("ollama", {}).get("models", [])

            if models:
                st.selectbox(
                    "Text model",
                    options=models,
                    key="ollama_model",
                    help="Model used for text extraction",
                )

                vision_models = [
                    m
                    for m in models
                    if "llava" in m.lower() or "vision" in m.lower() or "bakllava" in m.lower()
                ]
                if vision_models:
                    st.selectbox(
                        "Image model (optional)",
                        options=["None"] + vision_models,
                        key="ollama_vision_model",
                        help="Image model for direct extraction from images",
                    )
                else:
                    st.info("Vision models are not installed. Run: ollama pull llava")
            else:
                st.text_input(
                    "Model name",
                    value="llama3.2",
                    key="ollama_model_manual",
                    help="Example: llama3.2 or mistral",
                )
                st.text_input(
                    "Image model (optional)",
                    value="",
                    key="ollama_vision_model_manual",
                    placeholder="Example: llava",
                    help="Image model name for direct extraction",
                )

            st.text_input(
                "Ollama URL",
                value=st.session_state.config.ollama.base_url,
                key="ollama_url",
                help="Usually http://localhost:11434",
            )

        elif selected_provider == LLMProvider.LM_STUDIO:
            st.subheader("LM Studio settings")

            st.text_input(
                "Model name",
                value=st.session_state.config.lm_studio.model or "",
                key="lm_studio_model",
                placeholder="Leave empty to use the loaded model",
                help="The model currently loaded in LM Studio",
            )

            st.text_input(
                "Image model (optional)",
                value=st.session_state.config.lm_studio.vision_model or "",
                key="lm_studio_vision_model",
                placeholder="Example: llava-v1.6",
                help="Image model for direct extraction",
            )

            st.text_input(
                "LM Studio URL",
                value=st.session_state.config.lm_studio.base_url,
                key="lm_studio_url",
                help="Usually http://localhost:1234",
            )

        elif selected_provider == LLMProvider.DEEPSEEK:
            st.subheader("Deepseek settings")

            st.text_input(
                "API key",
                value=st.session_state.config.deepseek.api_key or "",
                key="deepseek_api_key",
                type="password",
                help="Your Deepseek API key",
            )

            st.text_input(
                "Model",
                value=st.session_state.config.deepseek.model,
                key="deepseek_model",
                help="Example: deepseek-chat",
            )

        st.divider()

        st.subheader("Scan quality")

        preprocessing_options = {
            PreprocessingLevel.NONE: "Fast (no cleanup)",
            PreprocessingLevel.LIGHT: "Light cleanup",
            PreprocessingLevel.STANDARD: "Standard (recommended)",
            PreprocessingLevel.AGGRESSIVE: "High cleanup (slow)",
        }

        st.selectbox(
            "Image cleanup",
            options=list(preprocessing_options.keys()),
            format_func=lambda x: preprocessing_options[x],
            key="preprocessing_level",
        )

        st.checkbox(
            "Use image understanding (skip text scan)",
            key="use_vision_model",
            help="Requires an image model in your AI engine",
        )

        st.divider()

        if st.button("Refresh system check"):
            st.session_state.system_validated = False
            st.rerun()


def render_header():
    """Render the main header and short description."""
    st.markdown(
        """
        <div class="app-header">
            <div style="font-size: 1.6rem; font-weight: 700;">Invoice Reader</div>
            <div class="muted">Upload an invoice, review the details, and save to Excel.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_progress_overview():
    """Render a lightweight progress overview of the workflow."""
    upload_ready = bool(st.session_state.uploaded_file_content)
    read_ready = bool(st.session_state.ocr_text or st.session_state.extracted_invoice)
    review_ready = bool(st.session_state.extracted_invoice)
    export_ready = bool(st.session_state.extracted_invoice)

    steps = [
        ("Add invoice", upload_ready),
        ("Read data", read_ready),
        ("Review", review_ready),
        ("Export", export_ready),
    ]

    pills = []
    for label, is_active in steps:
        state_class = "step-pill" if is_active else "step-pill inactive"
        pills.append(f"<span class=\"{state_class}\">{label}</span>")

    st.markdown(
        f"<div class=\"step-row\">{''.join(pills)}</div>",
        unsafe_allow_html=True,
    )


def render_help_panel():
    """Render a short help panel for new users."""
    st.subheader("Quick help")
    st.markdown(
        """
        <div class="help-callout">
            <strong>Start here</strong><br/>
            1. Add an invoice file.<br/>
            2. Click <em>Read invoice</em>.<br/>
            3. Review and correct the details.<br/>
            4. Save to Excel.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_section():
    """Render the file upload section."""
    st.subheader("Add an invoice")
    st.caption("Drag and drop a PDF or image. We use it only for this session.")

    uploaded_file = st.file_uploader(
        "Drag and drop or browse files",
        type=["pdf", "png", "jpg", "jpeg", "jpe", "jfif", "tiff", "tif", "bmp", "gif", "webp"],
        help="Supported: PDF, PNG, JPG, TIFF, BMP, GIF, WEBP",
    )
    
    if uploaded_file:
        st.session_state.uploaded_file_content = uploaded_file.read()
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.uploaded_file_type = uploaded_file.type
        
        st.success("File added. You can read it in the next step.")

        # Show preview for images
        if uploaded_file.type.startswith("image/"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(
                    st.session_state.uploaded_file_content,
                    caption="Invoice preview",
                    use_container_width=True,
                )
        
        return True
    
    return False


def perform_ocr():
    """Perform OCR on the uploaded file."""
    if not st.session_state.uploaded_file_content:
        st.error("No file uploaded")
        return None
    
    preprocessing_level = st.session_state.get(
        "preprocessing_level",
        PreprocessingLevel.STANDARD,
    )
    
    extractor = OCRExtractor(preprocessing_level=preprocessing_level)
    
    # Determine file extension
    file_name = st.session_state.uploaded_file_name
    extension = Path(file_name).suffix
    
    with st.spinner("Reading text from the invoice..."):
        result = extractor.extract(
            st.session_state.uploaded_file_content,
            file_extension=extension,
        )
    
    if result.errors:
        for error in result.errors:
            st.error(error)
        return None
    
    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)
    
    st.success(
        f"Text scan completed in {result.processing_time_seconds:.2f}s "
        f"(confidence: {result.confidence:.0%})"
    )
    
    return result.text


def perform_llm_extraction(ocr_text: Optional[str] = None, use_vision: bool = False):
    """Perform LLM extraction."""
    provider = st.session_state.get("selected_provider")
    if not provider:
        st.error("No LLM provider selected")
        return None
    
    config = st.session_state.config
    
    # Update config based on sidebar settings
    if provider == LLMProvider.OLLAMA:
        # Check for model from dropdown or manual entry
        if model := st.session_state.get("ollama_model"):
            config.ollama.model = model
        elif model := st.session_state.get("ollama_model_manual"):
            config.ollama.model = model
        
        # Check for vision model from dropdown or manual entry
        if vision_model := st.session_state.get("ollama_vision_model"):
            if vision_model != "None":
                config.ollama.vision_model = vision_model
        elif vision_model := st.session_state.get("ollama_vision_model_manual"):
            if vision_model:
                config.ollama.vision_model = vision_model
        
        # Update URL if changed
        if url := st.session_state.get("ollama_url"):
            config.ollama.base_url = url
            
    elif provider == LLMProvider.LM_STUDIO:
        if model := st.session_state.get("lm_studio_model"):
            config.lm_studio.model = model
        if vision_model := st.session_state.get("lm_studio_vision_model"):
            config.lm_studio.vision_model = vision_model
        if url := st.session_state.get("lm_studio_url"):
            config.lm_studio.base_url = url
            
    elif provider == LLMProvider.DEEPSEEK:
        if api_key := st.session_state.get("deepseek_api_key"):
            config.deepseek.api_key = api_key
        if model := st.session_state.get("deepseek_model"):
            config.deepseek.model = model
    
    client = LLMClient(config=config, preferred_provider=provider)
    parser = InvoiceParser()
    
    try:
        if use_vision and st.session_state.uploaded_file_content:
            with st.spinner(f"Reading with {provider.value} image model..."):
                response, used_provider = client.extract_from_image(
                    st.session_state.uploaded_file_content,
                    provider=provider,
                )
        else:
            if not ocr_text:
                st.error("No OCR text available")
                return None
            
            with st.spinner(f"Reading with {provider.value}..."):
                response, used_provider = client.extract_from_text(
                    ocr_text,
                    provider=provider,
                )
        
        # Parse response
        result = parser.parse_response(response)
        result.provider_used = used_provider.value
        
        if not result.success:
            for error in result.errors:
                st.error(error)
            return None
        
        if result.warnings:
            for warning in result.warnings:
                st.warning(warning)
        
        st.session_state.last_provider_used = used_provider.value
        st.success(f"Extraction completed using {used_provider.value}")
        
        return result
        
    except LLMClientError as e:
        st.error(f"AI extraction failed: {e}")
        return None


def render_extraction_section():
    """Render the extraction workflow section."""
    if not st.session_state.uploaded_file_content:
        st.info("Add an invoice first, then come back to read it.")
        return

    st.subheader("Read the invoice")
    st.caption("We scan the document and pull out the key details for review.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("<div class='primary-cta'>", unsafe_allow_html=True)
            run_extraction = st.button("Read invoice", type="primary", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if run_extraction:
            # Step 1: OCR
            ocr_text = perform_ocr()
            if ocr_text:
                st.session_state.ocr_text = ocr_text
                
                # Step 2: LLM extraction
                result = perform_llm_extraction(ocr_text)
                if result and result.invoice:
                    st.session_state.extracted_invoice = result.invoice
                    st.session_state.extraction_result = result
    
    with col2:
        use_vision = st.session_state.get("use_vision_model", False)
        vision_available = bool(st.session_state.config.ollama.vision_model)
        
        with st.container():
            st.markdown("<div class='secondary-cta'>", unsafe_allow_html=True)
            run_vision = st.button(
                "Use image understanding",
                use_container_width=True,
                disabled=not (use_vision and vision_available),
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if run_vision:
            result = perform_llm_extraction(use_vision=True)
            if result and result.invoice:
                st.session_state.extracted_invoice = result.invoice
                st.session_state.extraction_result = result

        if not use_vision:
            st.caption("Turn on image understanding in Expert mode if you want to skip text scanning.")
        elif use_vision and not vision_available:
            st.caption("Image model not found. Add one in Expert mode.")

    if st.session_state.last_provider_used:
        st.caption(f"Last run used: {st.session_state.last_provider_used}.")
    
    # Show OCR text if available
    if st.session_state.ocr_text:
        with st.expander("Raw text (advanced)", expanded=False):
            st.text_area(
                "Extracted text",
                value=st.session_state.ocr_text,
                height=300,
                disabled=True,
            )


def render_review_section():
    """Render the data review and edit section."""
    invoice = st.session_state.extracted_invoice
    if not invoice:
        st.info("Read an invoice to review the details here.")
        return
    
    st.subheader("Review and correct details")
    st.caption("Check the key fields and edit anything that looks wrong.")
    
    # Create editable form
    with st.form("invoice_edit_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Invoice details")
            invoice_number = st.text_input(
                "Invoice Number",
                value=invoice.header.invoice_number or "",
            )
            invoice_date = st.text_input(
                "Invoice Date",
                value=invoice.header.invoice_date or "",
            )
            due_date = st.text_input(
                "Due Date",
                value=invoice.header.due_date or "",
            )
            
            st.subheader("Seller")
            seller_name = st.text_input(
                "Seller Name",
                value=invoice.seller.name or "",
            )
            seller_vat = st.text_input(
                "Seller VAT Number",
                value=invoice.seller.vat_number or "",
            )
            seller_address = st.text_area(
                "Seller Address",
                value=invoice.seller.address or "",
                height=100,
            )
        
        with col2:
            st.subheader("Customer")
            customer_name = st.text_input(
                "Customer Name",
                value=invoice.customer.name or "",
            )
            customer_account = st.text_input(
                "Customer Account",
                value=invoice.customer.account_number or "",
            )
            customer_address = st.text_area(
                "Customer Address",
                value=invoice.customer.address or "",
                height=100,
            )
            
            st.subheader("Totals")
            subtotal = st.number_input(
                "Subtotal (R)",
                value=invoice.totals.subtotal or 0.0,
                format="%.2f",
            )
            vat_amount = st.number_input(
                "VAT Amount (R)",
                value=invoice.totals.vat_amount or 0.0,
                format="%.2f",
            )
            total_due = st.number_input(
                "Total Due (R)",
                value=invoice.totals.total_due or 0.0,
                format="%.2f",
            )
        
        # Line items
        st.subheader("Line items")
        
        line_items_data = []
        for i, item in enumerate(invoice.line_items):
            st.markdown(f"**Item {i + 1}**")
            cols = st.columns([3, 1, 1, 1, 1])
            
            with cols[0]:
                desc = st.text_input(
                    f"Description_{i}",
                    value=item.description,
                    label_visibility="collapsed",
                )
            with cols[1]:
                qty = st.number_input(
                    f"Qty_{i}",
                    value=item.quantity,
                    label_visibility="collapsed",
                )
            with cols[2]:
                price = st.number_input(
                    f"Price_{i}",
                    value=item.unit_price,
                    format="%.2f",
                    label_visibility="collapsed",
                )
            with cols[3]:
                total = st.number_input(
                    f"Total_{i}",
                    value=item.line_total or 0.0,
                    format="%.2f",
                    label_visibility="collapsed",
                )
            with cols[4]:
                vat_rate = st.number_input(
                    f"VAT_{i}",
                    value=item.vat_rate,
                    format="%.1f",
                    label_visibility="collapsed",
                )
            
            line_items_data.append({
                "description": desc,
                "quantity": qty,
                "unit_price": price,
                "line_total": total,
                "vat_rate": vat_rate,
                "unit_of_measure": item.unit_of_measure,
            })
        
        if st.form_submit_button("Save changes", type="primary"):
            # Update invoice with edited values
            invoice.header.invoice_number = invoice_number or None
            invoice.header.invoice_date = invoice_date or None
            invoice.header.due_date = due_date or None
            
            invoice.seller.name = seller_name or None
            invoice.seller.vat_number = seller_vat or None
            invoice.seller.address = seller_address or None
            
            invoice.customer.name = customer_name or None
            invoice.customer.account_number = customer_account or None
            invoice.customer.address = customer_address or None
            
            invoice.totals.subtotal = subtotal
            invoice.totals.vat_amount = vat_amount
            invoice.totals.total_due = total_due
            
            # Update line items
            invoice.line_items = [
                LineItem(**data) for data in line_items_data
            ]
            
            st.session_state.extracted_invoice = invoice
            st.success("Changes saved.")


def render_export_section():
    """Render the Excel export section."""
    invoice = st.session_state.extracted_invoice
    if not invoice:
        return
    
    st.subheader("Save to Excel")
    st.caption("Review the preview and choose where to save the file.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show preview
        st.subheader("Preview")
        rows = invoice.to_excel_rows()
        
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
    
    with col2:
        st.subheader("Save options")
        
        # File selection
        export_path = st.text_input(
            "File path",
            value=str(Path.home() / "Downloads" / "invoices.xlsx"),
            help="Full path to the Excel file",
        )

        st.checkbox(
            "Add to existing file",
            value=True,
            key="append_export",
            help="If unchecked, the file will be overwritten",
        )
        
        append = st.session_state.get("append_export", True)
        
        if st.button("Save to Excel", type="primary", use_container_width=True):
            try:
                exporter = ExcelExporter()
                result_path = exporter.export(invoice, export_path, append=append)
                st.session_state.export_result_path = result_path
                st.success(f"Saved to: {result_path}")
                
                # Offer download
                with open(result_path, "rb") as f:
                    st.download_button(
                        "Download Excel file",
                        data=f.read(),
                        file_name=Path(result_path).name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            except Exception as e:
                st.error(f"Save failed: {e}")
                logger.exception("Excel export failed")

        if st.session_state.export_result_path:
            st.caption(f"Last saved: {st.session_state.export_result_path}")


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Invoice Reader",
        page_icon="file",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    # Initialize
    apply_theme()
    init_session_state()
    
    # Header
    render_header()
    render_progress_overview()
    
    # Validate system on first load
    validate_system()

    # Main content layout
    main_col, side_col = st.columns([3, 1])

    with main_col:
        tabs = st.tabs([
            "1. Add invoice",
            "2. Read",
            "3. Review",
            "4. Export",
        ])

        with tabs[0]:
            render_upload_section()

        with tabs[1]:
            render_extraction_section()

        with tabs[2]:
            render_review_section()

        with tabs[3]:
            render_export_section()

    with side_col:
        render_help_panel()
        st.divider()
        render_system_status_panel()
        st.divider()
        render_expert_settings_panel()


if __name__ == "__main__":
    main()
