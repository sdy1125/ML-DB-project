# PDF knowledge ingestion

Luồng này scan trực tiếp bộ PDF local trong project, tách text thành chunk có metadata để index RAG/Qdrant. Phần text đọc tay được gom đúng 5 nhóm theo yêu cầu đồ án, không xuất rời rạc từng PDF nữa.

```text
rag/tailieuLLM-.../tailieuLLM/Nhóm 1..5/*.pdf
rag/tailieuLLM-.../tailieuLLM/*.pdf
        ↓
pypdf text extraction
        ↓
data/knowledge/processed/chunks.jsonl
data/knowledge/processed/parse_report.json
data/knowledge/text/group_1_imd_wef_reports.txt
data/knowledge/text/group_2_vietnam_country_reports.txt
data/knowledge/text/group_3_country_case_studies.txt
data/knowledge/text/group_4_scientific_papers.txt
data/knowledge/text/group_5_vietnam_policies.txt
        ↓
rag-indexer
        ↓
Qdrant
```

## Thư mục đầu vào hiện tại

Docker Compose đang mount thư mục này làm nguồn PDF:

```text
rag/tailieuLLM-20260626T154614Z-3-001/tailieuLLM
```

Bên trong có 5 nhóm chính:

- `Nhóm 1`: IMD & WEF Reports
- `Nhóm 2`: Vietnam Country Reports
- `Nhóm 3`: Case Studies nước tăng hạng IMD nhanh
- `Nhóm 4`: Academic Papers
- `Nhóm 5`: Chính sách Việt Nam

Các file PDF luật/chính sách Việt Nam đặt ở root của thư mục `tailieuLLM` cũng được nhập vào `Nhóm 5`. Project không dùng group `Root` nữa.

## Metadata tự sinh

Parser tự infer metadata từ folder:

| Folder | Metadata `group` | `source_type` |
|---|---|---|
| `Nhóm 1` | `group_1_imd_wef_reports` | `competitiveness_report` |
| `Nhóm 2` | `group_2_vietnam_country_reports` | `country_report` |
| `Nhóm 3` | `group_3_country_case_studies` | `case_study` |
| `Nhóm 4` | `group_4_scientific_papers` | `scientific_paper` |
| `Nhóm 5` hoặc PDF root | `group_5_vietnam_policies` | `policy_document` |

Mỗi chunk trong `chunks.jsonl` có các field chính:

- `document_id`
- `source`
- `source_path`
- `title`
- `group`
- `group_label`
- `collection_role`
- `country`
- `source_type`
- `page_start`, `page_end`
- `element_categories`
- `chunk_index`
- `char_count`

Nếu file match được `configs/knowledge_manifest.yaml`, metadata trong manifest sẽ được merge thêm. Nếu không match, parser vẫn chạy bình thường bằng metadata infer từ đường dẫn.

## Chạy parser

```powershell
docker compose --profile jobs run --rm pdf-parser
```

Hoặc chạy local:

```powershell
python -m rag.parse_pdfs --input-dir "rag\tailieuLLM-20260626T154614Z-3-001\tailieuLLM" --manifest configs\knowledge_manifest.yaml --output data\knowledge\processed\chunks.jsonl --text-output-dir data\knowledge\text
```

Output chính:

- `data/knowledge/processed/chunks.jsonl`: dữ liệu nhỏ theo chunk cho RAG.
- `data/knowledge/processed/parse_report.json`: báo cáo parse từng PDF.
- `data/knowledge/text/group_*.txt`: 5 file đọc tay theo 5 nhóm.

Nếu đã có `chunks.jsonl` và chỉ muốn dựng lại 5 file text:

```powershell
python -m rag.parse_pdfs --output data\knowledge\processed\chunks.jsonl --text-output-dir data\knowledge\text --export-only
```

## Index vào Qdrant

Sau khi cấu hình embedding:

```powershell
docker compose --profile jobs run --rm rag-indexer
```

## Chiến lược parse

Mặc định dùng parser nhẹ `pypdf`:

```dotenv
PDF_PARSE_STRATEGY=pypdf
PDF_TEXT_EXPORT_MODE=group
```

Lý do: `unstructured[pdf]` kéo dependency nặng như ONNX, Poppler/Tesseract và dễ lỗi build Docker nếu thiếu CMake. Với bộ PDF text-based hiện tại, `pypdf` đủ để dựng trước kho tri thức. Sau này nếu gặp PDF scan ảnh hoặc bảng phức tạp, có thể thêm pipeline OCR riêng rồi bật lại `fast`, `hi_res` hoặc `ocr_only`.
