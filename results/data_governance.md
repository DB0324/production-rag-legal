# Data Governance Notes

## Dataset Licensing
- **KanoonGPT Indian Case Laws** (Hugging Face): Supreme Court judgments
  are public court records, published for public access. Used here for
  a non-commercial academic/portfolio project.
- **IndicLegalQA Dataset** (Mendeley Data, DOI 10.17632/gf8n8cnmvc.2):
  CC BY 4.0 licensed -- permits reuse with attribution. Attribution:
  K, Veningston; Mishra, Apratim. "IndicLegalQA Dataset." Mendeley Data,
  V2, 2024.

## PII / Sensitive Data
- Source documents are public Supreme Court judgments -- party names,
  case details, and judicial reasoning are part of the public record by
  design (open justice principle, as directly discussed in one of the
  corpus documents: Swapnil Tripathi v. Supreme Court of India, 2018,
  on public access to court proceedings).
- No private/internal/confidential data is used anywhere in this project.
- Application logs (results/*.json) contain only questions, generated
  answers, and citations to public judgments -- no user-identifying
  information is collected or logged, since this is a demo system with
  no real end-user traffic.

## If Extended Beyond This Project
Were this pipeline extended to a domain with actual PII (e.g. real
internal company documents, medical records), the following would be
required before production use:
- PII redaction/anonymization at the ingestion stage
- Access controls on the vector DB and API layer
- Log retention policy and encryption at rest
- Audit trail for who queried what and when

None of the above was necessary for this project's public-legal-data
scope, but is noted here to show awareness of the gap between "portfolio
demo" and "handles real sensitive data."
