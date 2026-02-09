"""
LLM response parser for invoice extraction.

Handles:
- JSON extraction from LLM responses
- Validation of extracted data
- Regex fallback for common patterns
- Error recovery
"""

import json
import logging
import re
from typing import Any, Optional

from app.models.invoice import (
    ExtractedInvoice,
    ExtractionResult,
    InvoiceHeader,
    InvoiceTotals,
    LineItem,
    SellerInfo,
    CustomerInfo,
)

logger = logging.getLogger(__name__)


class InvoiceParser:
    """
    Parses LLM responses and extracts invoice data.
    
    Handles various response formats and includes fallback
    extraction using regex patterns.
    """
    
    # Regex patterns for fallback extraction
    PATTERNS = {
        "invoice_number": [
            r"(?:invoice|inv|tax invoice)[\s#:]*(\d+)",
            r"#(\d{4,})",
            r"invoice number[:\s]*(\S+)",
        ],
        "date": [
            r"(\d{4}-\d{2}-\d{2})",  # ISO format
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",  # Various date formats
            r"date[:\s]*(\d{1,2}\s+\w+\s+\d{4})",  # 15 January 2024
        ],
        "vat_number": [
            r"vat[\s#:]*(\d{10})",
            r"vat\s*(?:no|number|reg)[.:\s]*(\d{10})",
        ],
        "total": [
            r"total[:\s]*R?\s*([\d,]+\.?\d*)",
            r"amount\s*due[:\s]*R?\s*([\d,]+\.?\d*)",
            r"balance[:\s]*R?\s*([\d,]+\.?\d*)",
        ],
        "vat_amount": [
            r"vat\s*(?:\(?\d+%?\)?)?[:\s]*R?\s*([\d,]+\.?\d*)",
            r"tax[:\s]*R?\s*([\d,]+\.?\d*)",
        ],
        "subtotal": [
            r"sub\s*total[:\s]*R?\s*([\d,]+\.?\d*)",
            r"excl\.?\s*vat[:\s]*R?\s*([\d,]+\.?\d*)",
        ],
    }
    
    def parse_response(self, response: str) -> ExtractionResult:
        """
        Parse LLM response and extract invoice data.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            ExtractionResult with parsed data or errors
        """
        errors = []
        warnings = []
        
        # Try to extract JSON from response
        json_str = self._extract_json(response)
        
        if not json_str:
            # Show truncated raw response for debugging
            preview = response[:500] if len(response) > 500 else response
            errors.append(f"No valid JSON found in response. Raw output: {preview}")
            return ExtractionResult(
                success=False,
                invoice=ExtractedInvoice(),
                ocr_text=response,
                errors=errors,
                warnings=warnings,
            )
        
        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            errors.append(f"JSON parse error: {str(e)}")
            return ExtractionResult(
                success=False,
                invoice=ExtractedInvoice(),
                ocr_text=response,
                errors=errors,
                warnings=warnings,
            )
        
        # Convert to typed models
        try:
            invoice = self._dict_to_invoice(data, warnings)
            
            # Validate the extraction
            validation_warnings = self._validate_extraction(invoice)
            warnings.extend(validation_warnings)
            
            return ExtractionResult(
                success=True,
                invoice=invoice,
                ocr_text=response,
                llm_provider="",  # Will be set by caller
                errors=errors,
                warnings=warnings,
            )
            
        except Exception as e:
            errors.append(f"Failed to parse invoice data: {str(e)}")
            logger.exception("Invoice parsing failed")
            return ExtractionResult(
                success=False,
                invoice=ExtractedInvoice(),
                ocr_text=response,
                errors=errors,
                warnings=warnings,
            )
    
    def _extract_json(self, response: str) -> Optional[str]:
        """Extract JSON object from response string."""
        # Strip qwen3 <think>...</think> reasoning blocks
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()
        # Also strip incomplete think blocks (model may not close the tag)
        cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned).strip()
        
        # Try to find JSON block in markdown code blocks
        code_block_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
        match = re.search(code_block_pattern, cleaned)
        if match:
            try:
                json.loads(match.group(1))
                return match.group(1)
            except json.JSONDecodeError:
                pass
        
        # Try the cleaned response as-is (may be pure JSON)
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            pass
        
        # Try to find the outermost JSON object with balanced braces
        start = cleaned.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == "{":
                    depth += 1
                elif cleaned[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start:i+1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            break
        
        # Try greedy regex as last resort
        json_pattern = r"(\{[\s\S]*\})"
        match = re.search(json_pattern, cleaned)
        if match:
            try:
                json.loads(match.group(1))
                return match.group(1)
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _dict_to_invoice(self, data: dict, warnings: list) -> ExtractedInvoice:
        """Convert dictionary to typed invoice model."""
        # Parse header
        header_data = data.get("header", {})
        header = InvoiceHeader(
            invoice_number=header_data.get("invoice_number") or "",
            invoice_date=header_data.get("invoice_date"),
            due_date=header_data.get("due_date"),
            order_number=header_data.get("purchase_order") or header_data.get("order_number") or "",
        )
        
        # Parse seller
        seller_data = data.get("seller", {})
        seller = SellerInfo(
            name=seller_data.get("name") or "",
            address=seller_data.get("address") or "",
            vat_number=seller_data.get("vat_number") or "",
            telephone=seller_data.get("phone") or seller_data.get("telephone") or "",
            email=seller_data.get("email") or "",
            registration_number=seller_data.get("registration_number") or "",
        )
        
        # Parse customer
        customer_data = data.get("customer", {})
        customer = CustomerInfo(
            name=customer_data.get("name") or "",
            address=customer_data.get("address") or "",
            account_number=customer_data.get("account_number") or "",
        )
        
        # Parse line items
        line_items = []
        for item_data in data.get("line_items", []):
            try:
                quantity = self._parse_number(item_data.get("quantity"), 1.0)
                unit_price = self._parse_number(item_data.get("unit_price"), 0.0)
                line_total = self._parse_number(item_data.get("line_total"))
                vat_rate = self._parse_number(item_data.get("vat_rate"), 15.0)
                
                # Calculate vat_amount from vat_rate if not provided directly
                vat_amount = self._parse_number(item_data.get("vat_amount"))
                if vat_amount is None and vat_rate and quantity and unit_price:
                    subtotal = quantity * unit_price
                    vat_amount = subtotal * (vat_rate / 100.0)
                
                item = LineItem(
                    item_code=item_data.get("item_code") or "",
                    description=item_data.get("description", "Unknown"),
                    quantity=quantity,
                    unit=item_data.get("unit_of_measure") or item_data.get("unit") or "each",
                    unit_price=unit_price,
                    vat_amount=vat_amount or 0.0,
                    line_total=line_total or 0.0,
                )
                line_items.append(item)
            except Exception as e:
                warnings.append(f"Failed to parse line item: {e}")
        
        # Parse totals
        totals_data = data.get("totals", {})
        totals = InvoiceTotals(
            subtotal_excl_vat=self._parse_number(
                totals_data.get("subtotal") or totals_data.get("subtotal_excl_vat"), 0.0
            ),
            total_vat=self._parse_number(
                totals_data.get("vat_amount") or totals_data.get("total_vat"), 0.0
            ),
            total_incl_vat=self._parse_number(
                totals_data.get("total_due") or totals_data.get("total_incl_vat"), 0.0
            ),
        )
        
        # Get metadata
        confidence = self._parse_number(data.get("confidence_score"), 0.5)
        notes = data.get("extraction_notes", [])
        
        return ExtractedInvoice(
            header=header,
            seller=seller,
            customer=customer,
            line_items=line_items,
            totals=totals,
            reference_notes="; ".join(notes) if isinstance(notes, list) else str(notes or ""),
        )
    
    def _parse_number(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        """Parse a number from various formats."""
        if value is None:
            return default
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove currency symbols and whitespace
            cleaned = re.sub(r"[R$€£,\s]", "", value)
            try:
                return float(cleaned)
            except ValueError:
                return default
        
        return default
    
    def _validate_extraction(self, invoice: ExtractedInvoice) -> list[str]:
        """Validate extracted data and return warnings."""
        warnings = []
        
        # Check for missing critical fields
        if not invoice.header.invoice_number:
            warnings.append("Missing invoice number")
        
        if not invoice.header.invoice_date:
            warnings.append("Missing invoice date")
        
        if not invoice.seller.name:
            warnings.append("Missing seller name")
        
        if len(invoice.line_items) == 0:
            warnings.append("No line items extracted")
        
        # Validate totals calculation
        if invoice.totals.subtotal and invoice.totals.vat_amount and invoice.totals.total_due:
            calculated_total = invoice.totals.subtotal + invoice.totals.vat_amount
            if abs(calculated_total - invoice.totals.total_due) > 1.0:  # Allow R1 tolerance
                warnings.append(
                    f"Total mismatch: subtotal ({invoice.totals.subtotal}) + "
                    f"VAT ({invoice.totals.vat_amount}) != "
                    f"total ({invoice.totals.total_due})"
                )
        
        # Validate line items sum
        if invoice.line_items and invoice.totals.subtotal:
            items_sum = sum(
                item.line_total for item in invoice.line_items
                if item.line_total is not None
            )
            if items_sum > 0 and abs(items_sum - invoice.totals.subtotal) > 1.0:
                warnings.append(
                    f"Line items sum ({items_sum:.2f}) doesn't match "
                    f"subtotal ({invoice.totals.subtotal:.2f})"
                )
        
        # Validate VAT number format (South African)
        if invoice.seller.vat_number:
            vat = invoice.seller.vat_number.replace(" ", "")
            if not (vat.isdigit() and len(vat) == 10 and vat.startswith("4")):
                warnings.append(f"VAT number '{invoice.seller.vat_number}' may be invalid (SA format: 10 digits starting with 4)")
        
        return warnings
    
    def extract_with_regex(self, ocr_text: str) -> dict:
        """
        Fallback extraction using regex patterns.
        
        Use this when LLM extraction fails or for validation.
        
        Args:
            ocr_text: Raw OCR text
            
        Returns:
            Dictionary with extracted fields
        """
        text_lower = ocr_text.lower()
        results = {}
        
        for field, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    results[field] = match.group(1)
                    break
        
        # Extract line items (more complex pattern)
        line_items = self._extract_line_items_regex(ocr_text)
        if line_items:
            results["line_items"] = line_items
        
        return results
    
    def _extract_line_items_regex(self, text: str) -> list[dict]:
        """
        Attempt to extract line items using regex.
        
        This is a fallback method that may not capture all items.
        """
        items = []
        
        # Pattern for: quantity x description @ price = total
        pattern1 = r"(\d+)\s*x\s*(.+?)\s*@\s*R?\s*([\d,]+\.?\d*)\s*=?\s*R?\s*([\d,]+\.?\d*)"
        
        for match in re.finditer(pattern1, text, re.IGNORECASE):
            try:
                items.append({
                    "quantity": float(match.group(1)),
                    "description": match.group(2).strip(),
                    "unit_price": float(match.group(3).replace(",", "")),
                    "line_total": float(match.group(4).replace(",", "")),
                })
            except ValueError:
                continue
        
        # Pattern for: description | qty | price | total (table format)
        pattern2 = r"([A-Za-z][A-Za-z\s]+?)\s+(\d+)\s+R?\s*([\d,]+\.?\d*)\s+R?\s*([\d,]+\.?\d*)"
        
        for match in re.finditer(pattern2, text):
            try:
                desc = match.group(1).strip()
                # Skip if it looks like a header
                if desc.lower() in ["description", "item", "product", "qty", "quantity"]:
                    continue
                items.append({
                    "description": desc,
                    "quantity": float(match.group(2)),
                    "unit_price": float(match.group(3).replace(",", "")),
                    "line_total": float(match.group(4).replace(",", "")),
                })
            except ValueError:
                continue
        
        return items


def parse_llm_response(response: str) -> ExtractionResult:
    """
    Convenience function to parse an LLM response.
    
    Args:
        response: Raw LLM response
        
    Returns:
        ExtractionResult with parsed data
    """
    parser = InvoiceParser()
    return parser.parse_response(response)
