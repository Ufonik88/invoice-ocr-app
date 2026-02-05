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


def validate_system():
    """Validate system requirements on startup."""
    if not st.session_state.system_validated:
        with st.spinner("Validating system requirements..."):
            results = validate_system_requirements()
            st.session_state.validation_results = results
            st.session_state.system_validated = True
    
    return st.session_state.validation_results


def show_validation_status():
    """Display system validation status."""
    results = st.session_state.validation_results
    if not results:
        return
    
    # Check for critical issues
    critical_issues = []
    warnings = []
    
    if not results["tesseract"]["installed"]:
        critical_issues.append(
            "⚠️ **Tesseract not found!** OCR will not work.\n\n"
            "Install with:\n"
            "- macOS: `brew install tesseract`\n"
            "- Ubuntu: `sudo apt install tesseract-ocr`\n"
            "- Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)"
        )
    
    if not results["ollama"]["available"] and not results["deepseek"]["configured"]:
        warnings.append(
            "⚠️ No LLM provider available. Start Ollama or configure Deepseek API key."
        )
    
    if critical_issues:
        for issue in critical_issues:
            st.error(issue)
    
    if warnings:
        for warning in warnings:
            st.warning(warning)
    
    # Show available providers
    available = []
    if results["ollama"]["available"]:
        models = results["ollama"].get("models", [])
        available.append(f"✅ Ollama ({len(models)} models)")
    if results["lm_studio"]["available"]:
        available.append("✅ LM Studio")
    if results["deepseek"]["configured"]:
        available.append("✅ Deepseek (cloud)")
    
    if available:
        st.success(f"Available providers: {', '.join(available)}")


def render_sidebar():
    """Render the settings sidebar."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # LLM Provider selection
        st.subheader("LLM Provider")
        
        results = st.session_state.validation_results or {}
        
        # Always show all providers, indicate which are available
        provider_names = {
            LLMProvider.OLLAMA: "Ollama (Local)",
            LLMProvider.LM_STUDIO: "LM Studio (Local)",
            LLMProvider.DEEPSEEK: "Deepseek (Cloud)",
        }
        
        all_providers = [LLMProvider.OLLAMA, LLMProvider.LM_STUDIO, LLMProvider.DEEPSEEK]
        
        selected_provider = st.selectbox(
            "Select Provider",
            options=all_providers,
            format_func=lambda x: provider_names.get(x, x.value),
            key="provider_selection",
        )
        st.session_state.selected_provider = selected_provider
        
        # Show provider status
        if selected_provider == LLMProvider.OLLAMA:
            if results.get("ollama", {}).get("available"):
                st.success("✅ Ollama is running")
            else:
                st.warning("⚠️ Ollama not running. Start with: `ollama serve`")
        elif selected_provider == LLMProvider.LM_STUDIO:
            if results.get("lm_studio", {}).get("available"):
                st.success("✅ LM Studio is running")
            else:
                st.warning("⚠️ LM Studio not running. Start the local server.")
        elif selected_provider == LLMProvider.DEEPSEEK:
            if results.get("deepseek", {}).get("configured"):
                st.success("✅ Deepseek API configured")
            else:
                st.warning("⚠️ Deepseek API key not set in .env")
        
        st.divider()
        
        # Provider-specific settings
        if selected_provider == LLMProvider.OLLAMA:
            st.subheader("Ollama Settings")
            
            # Get models from validation or allow manual entry
            models = results.get("ollama", {}).get("models", [])
            
            if models:
                st.selectbox(
                    "Text Model",
                    options=models,
                    key="ollama_model",
                    help="Select the model to use for text extraction",
                )
                
                # Vision model option
                vision_models = [m for m in models if "llava" in m.lower() or "vision" in m.lower() or "bakllava" in m.lower()]
                if vision_models:
                    st.selectbox(
                        "Vision Model (optional)",
                        options=["None"] + vision_models,
                        key="ollama_vision_model",
                        help="Select a vision model for direct image extraction",
                    )
                else:
                    st.info("💡 For vision support, run: `ollama pull llava`")
            else:
                # Manual model entry when Ollama not running
                st.text_input(
                    "Model Name",
                    value="llama3.2",
                    key="ollama_model_manual",
                    help="Enter the model name (e.g., llama3.2, mistral, etc.)",
                )
                st.text_input(
                    "Vision Model (optional)",
                    value="",
                    key="ollama_vision_model_manual",
                    placeholder="e.g., llava",
                    help="Enter a vision model name for direct image extraction",
                )
            
            # Ollama URL config
            st.text_input(
                "Ollama URL",
                value=st.session_state.config.ollama.base_url,
                key="ollama_url",
                help="Usually http://localhost:11434",
            )
            
        elif selected_provider == LLMProvider.LM_STUDIO:
            st.subheader("LM Studio Settings")
            
            st.text_input(
                "Model Name",
                value=st.session_state.config.lm_studio.model or "",
                key="lm_studio_model",
                placeholder="Leave empty to use loaded model",
                help="The model loaded in LM Studio",
            )
            
            st.text_input(
                "Vision Model (optional)",
                value=st.session_state.config.lm_studio.vision_model or "",
                key="lm_studio_vision_model",
                placeholder="e.g., llava-v1.6",
                help="Vision model for direct image extraction",
            )
            
            st.text_input(
                "LM Studio URL",
                value=st.session_state.config.lm_studio.base_url,
                key="lm_studio_url",
                help="Usually http://localhost:1234",
            )
            
        elif selected_provider == LLMProvider.DEEPSEEK:
            st.subheader("Deepseek Settings")
            
            st.text_input(
                "API Key",
                value=st.session_state.config.deepseek.api_key or "",
                key="deepseek_api_key",
                type="password",
                help="Your Deepseek API key",
            )
            
            st.text_input(
                "Model",
                value=st.session_state.config.deepseek.model,
                key="deepseek_model",
                help="e.g., deepseek-chat",
            )
        
        st.divider()
        
        # OCR Settings
        st.subheader("OCR Settings")
        
        preprocessing_options = {
            PreprocessingLevel.NONE: "None (fastest)",
            PreprocessingLevel.LIGHT: "Light (basic)",
            PreprocessingLevel.STANDARD: "Standard (recommended)",
            PreprocessingLevel.AGGRESSIVE: "Aggressive (for poor scans)",
        }
        
        st.selectbox(
            "Preprocessing Level",
            options=list(preprocessing_options.keys()),
            format_func=lambda x: preprocessing_options[x],
            key="preprocessing_level",
        )
        
        st.checkbox(
            "Use Vision Model (skip OCR)",
            key="use_vision_model",
            help="If enabled and a vision model is available, extract directly from image",
        )
        
        st.divider()
        
        # Export Settings
        st.subheader("Export Settings")
        
        st.checkbox(
            "Append to existing file",
            value=True,
            key="append_export",
            help="If unchecked, will overwrite the file",
        )
        
        st.divider()
        
        # System status
        st.subheader("System Status")
        
        if st.button("🔄 Refresh Status"):
            st.session_state.system_validated = False
            st.rerun()
        
        results = st.session_state.validation_results or {}
        
        tesseract = results.get("tesseract", {})
        if tesseract.get("installed"):
            st.success(f"✅ Tesseract {tesseract.get('version', 'installed')}")
        else:
            st.error("❌ Tesseract not found")


def render_upload_section():
    """Render the file upload section."""
    st.header("📄 Upload Invoice")
    
    uploaded_file = st.file_uploader(
        "Choose an invoice file",
        type=["pdf", "png", "jpg", "jpeg", "jpe", "jfif", "tiff", "tif", "bmp", "gif", "webp"],
        help="Upload a PDF or image file of your invoice (supports PDF, PNG, JPG, TIFF, BMP, GIF, WEBP)",
    )
    
    if uploaded_file:
        st.session_state.uploaded_file_content = uploaded_file.read()
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.uploaded_file_type = uploaded_file.type
        
        # Show preview for images
        if uploaded_file.type.startswith("image/"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(
                    st.session_state.uploaded_file_content,
                    caption="Uploaded Invoice",
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
    
    with st.spinner("Extracting text with OCR..."):
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
        f"OCR completed in {result.processing_time_seconds:.2f}s "
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
            with st.spinner(f"Extracting with {provider.value} vision model..."):
                response, used_provider = client.extract_from_image(
                    st.session_state.uploaded_file_content,
                    provider=provider,
                )
        else:
            if not ocr_text:
                st.error("No OCR text available")
                return None
            
            with st.spinner(f"Extracting with {provider.value}..."):
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
        
        st.success(f"Extraction completed using {used_provider.value}")
        
        return result
        
    except LLMClientError as e:
        st.error(f"LLM extraction failed: {e}")
        return None


def render_extraction_section():
    """Render the extraction workflow section."""
    st.header("🔍 Extract Invoice Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Run OCR + LLM Extraction", type="primary", use_container_width=True):
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
        
        if st.button(
            "👁️ Vision Extraction (Skip OCR)",
            use_container_width=True,
            disabled=not (use_vision and vision_available),
        ):
            result = perform_llm_extraction(use_vision=True)
            if result and result.invoice:
                st.session_state.extracted_invoice = result.invoice
                st.session_state.extraction_result = result
    
    # Show OCR text if available
    if st.session_state.ocr_text:
        with st.expander("📄 OCR Text", expanded=False):
            st.text_area(
                "Extracted Text",
                value=st.session_state.ocr_text,
                height=300,
                disabled=True,
            )


def render_review_section():
    """Render the data review and edit section."""
    invoice = st.session_state.extracted_invoice
    if not invoice:
        st.info("Extract invoice data to review and edit")
        return
    
    st.header("✏️ Review & Edit Data")
    
    # Create editable form
    with st.form("invoice_edit_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Invoice Header")
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
            
            st.subheader("Seller Information")
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
            st.subheader("Customer Information")
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
        st.subheader("Line Items")
        
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
        
        if st.form_submit_button("💾 Save Changes", type="primary"):
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
            st.success("Changes saved!")


def render_export_section():
    """Render the Excel export section."""
    invoice = st.session_state.extracted_invoice
    if not invoice:
        return
    
    st.header("📊 Export to Excel")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show preview
        st.subheader("Export Preview")
        rows = invoice.to_excel_rows()
        
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
    
    with col2:
        st.subheader("Export Options")
        
        # File selection
        export_path = st.text_input(
            "Export File Path",
            value=str(Path.home() / "Downloads" / "invoices.xlsx"),
            help="Full path to the Excel file",
        )
        
        append = st.session_state.get("append_export", True)
        
        if st.button("📥 Export to Excel", type="primary", use_container_width=True):
            try:
                exporter = ExcelExporter()
                result_path = exporter.export(invoice, export_path, append=append)
                st.success(f"✅ Exported to: {result_path}")
                
                # Offer download
                with open(result_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Excel File",
                        data=f.read(),
                        file_name=Path(result_path).name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            except Exception as e:
                st.error(f"Export failed: {e}")
                logger.exception("Excel export failed")


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Invoice OCR Extractor",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Initialize
    init_session_state()
    
    # Header
    st.title("📄 Invoice OCR Extractor")
    st.markdown(
        "Extract data from South African tax invoices using OCR and AI. "
        "Export to Excel with ZAR formatting."
    )
    
    # Validate system on first load
    validate_system()
    show_validation_status()
    
    # Sidebar
    render_sidebar()
    
    # Main content
    st.divider()
    
    # File upload
    if render_upload_section():
        st.divider()
        
        # Extraction
        render_extraction_section()
        
        st.divider()
        
        # Review
        render_review_section()
        
        st.divider()
        
        # Export
        render_export_section()


if __name__ == "__main__":
    main()
