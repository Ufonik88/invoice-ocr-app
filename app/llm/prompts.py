"""
LLM prompt templates for invoice data extraction.

Contains optimized prompts for:
- Text-based extraction (from OCR output)
- Vision-based extraction (direct from image)
- South African invoice format handling
"""

# Few-shot example based on sample invoice structure
FEW_SHOT_EXAMPLE = '''
Example Invoice Text:
---
WC Fresh Beverages
123 Main Street, Cape Town
VAT No: 4123456789
Tax Invoice #12345
Date: 2024-01-15
Customer: ABC Company
Line Items:
2 x Widget A @ R100.00 = R200.00
5 x Widget B @ R50.00 = R250.00
Subtotal: R450.00
VAT (15%): R67.50
Total: R517.50
---

Expected JSON Output:
{
  "header": {
    "invoice_number": "12345",
    "invoice_date": "2024-01-15",
    "due_date": null,
    "purchase_order": null
  },
  "seller": {
    "name": "WC Fresh Beverages",
    "address": "123 Main Street, Cape Town",
    "vat_number": "4123456789",
    "phone": null,
    "email": null
  },
  "customer": {
    "name": "ABC Company",
    "address": null,
    "account_number": null
  },
  "line_items": [
    {
      "description": "Widget A",
      "quantity": 2.0,
      "unit_price": 100.00,
      "line_total": 200.00,
      "vat_rate": 15.0,
      "unit_of_measure": "each"
    },
    {
      "description": "Widget B",
      "quantity": 5.0,
      "unit_price": 50.00,
      "line_total": 250.00,
      "vat_rate": 15.0,
      "unit_of_measure": "each"
    }
  ],
  "totals": {
    "subtotal": 450.00,
    "vat_amount": 67.50,
    "total_due": 517.50,
    "discount_amount": 0.00
  },
  "confidence_score": 0.95,
  "extraction_notes": ["All fields extracted successfully"]
}
'''


INVOICE_EXTRACTION_PROMPT = '''You are an expert invoice data extractor specializing in South African tax invoices.

Your task is to extract structured data from the OCR text of an invoice and return it as a valid JSON object.

IMPORTANT RULES:
1. Extract ALL line items individually - each product/service should be a separate entry
2. Currency is South African Rand (ZAR/R). Remove currency symbols when storing numeric values
3. VAT in South Africa is 15% - use this if not explicitly stated
4. Dates should be in ISO format (YYYY-MM-DD)
5. If a value cannot be found or is unclear, use null
6. VAT numbers in SA typically start with 4 and are 10 digits
7. Be careful with OCR errors - common issues include:
   - 0/O confusion
   - 1/l/I confusion  
   - Misaligned columns
   - Missing decimal points

REQUIRED JSON STRUCTURE:
{
  "header": {
    "invoice_number": "string or null",
    "invoice_date": "YYYY-MM-DD or null",
    "due_date": "YYYY-MM-DD or null",
    "purchase_order": "string or null"
  },
  "seller": {
    "name": "string or null",
    "address": "string or null",
    "vat_number": "string or null",
    "phone": "string or null",
    "email": "string or null"
  },
  "customer": {
    "name": "string or null",
    "address": "string or null",
    "account_number": "string or null"
  },
  "line_items": [
    {
      "description": "string",
      "quantity": number,
      "unit_price": number,
      "line_total": number,
      "vat_rate": 15.0,
      "unit_of_measure": "string or null"
    }
  ],
  "totals": {
    "subtotal": number or null,
    "vat_amount": number or null,
    "total_due": number or null,
    "discount_amount": number or null
  },
  "confidence_score": 0.0 to 1.0,
  "extraction_notes": ["list of any issues or uncertainties"]
}

{few_shot}

Now extract data from this invoice:
---
{ocr_text}
---

Return ONLY the JSON object, no additional text or markdown formatting.
'''


VISION_EXTRACTION_PROMPT = '''You are an expert invoice data extractor with vision capabilities, specializing in South African tax invoices.

Analyze this invoice image and extract all relevant data into a structured JSON format.

IMPORTANT RULES:
1. Extract ALL line items individually - each product/service should be a separate entry
2. Currency is South African Rand (ZAR/R). Store numeric values without currency symbols
3. VAT in South Africa is 15% - use this if not explicitly stated
4. Dates should be in ISO format (YYYY-MM-DD)
5. If a value cannot be determined, use null
6. VAT numbers in SA typically start with 4 and are 10 digits
7. Pay attention to:
   - Table structures and column alignments
   - Header information at the top
   - Totals section usually at the bottom
   - Any stamps, logos, or watermarks that contain business info

REQUIRED JSON STRUCTURE:
{
  "header": {
    "invoice_number": "string or null",
    "invoice_date": "YYYY-MM-DD or null", 
    "due_date": "YYYY-MM-DD or null",
    "purchase_order": "string or null"
  },
  "seller": {
    "name": "string or null",
    "address": "string or null",
    "vat_number": "string or null",
    "phone": "string or null",
    "email": "string or null"
  },
  "customer": {
    "name": "string or null",
    "address": "string or null",
    "account_number": "string or null"
  },
  "line_items": [
    {
      "description": "string",
      "quantity": number,
      "unit_price": number,
      "line_total": number,
      "vat_rate": 15.0,
      "unit_of_measure": "string or null"
    }
  ],
  "totals": {
    "subtotal": number or null,
    "vat_amount": number or null,
    "total_due": number or null,
    "discount_amount": number or null
  },
  "confidence_score": 0.0 to 1.0,
  "extraction_notes": ["list of any issues or observations"]
}

Return ONLY the JSON object, no additional text or markdown formatting.
'''


VALIDATION_PROMPT = '''You are validating extracted invoice data for accuracy.

Compare the extracted JSON data against the original OCR text and:
1. Check if all line items were captured
2. Verify numeric calculations (subtotal + VAT = total)
3. Flag any suspicious values or potential OCR errors
4. Suggest corrections if needed

Original OCR Text:
---
{ocr_text}
---

Extracted Data:
---
{extracted_json}
---

Provide your validation result as JSON:
{
  "is_valid": true/false,
  "issues": ["list of issues found"],
  "suggested_corrections": {
    "field_path": "corrected_value"
  },
  "validation_notes": "any additional observations"
}
'''


def get_extraction_prompt(
    ocr_text: str,
    include_few_shot: bool = True,
) -> str:
    """
    Generate the extraction prompt with OCR text.
    
    Args:
        ocr_text: The OCR-extracted text from the invoice
        include_few_shot: Whether to include the few-shot example
        
    Returns:
        Formatted prompt string
    """
    few_shot = FEW_SHOT_EXAMPLE if include_few_shot else ""
    return INVOICE_EXTRACTION_PROMPT.format(
        ocr_text=ocr_text,
        few_shot=few_shot,
    )


def get_vision_prompt() -> str:
    """
    Get the vision extraction prompt.
    
    Returns:
        Vision prompt string
    """
    return VISION_EXTRACTION_PROMPT


def get_validation_prompt(ocr_text: str, extracted_json: str) -> str:
    """
    Generate the validation prompt.
    
    Args:
        ocr_text: Original OCR text
        extracted_json: Extracted data as JSON string
        
    Returns:
        Formatted validation prompt
    """
    return VALIDATION_PROMPT.format(
        ocr_text=ocr_text,
        extracted_json=extracted_json,
    )
