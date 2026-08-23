# Face Verification & KYC System
## Version-Wise Product and Technical Roadmap

### Technology Stack

**Frontend**
- React
- TypeScript
- Vite

**Backend**
- Python
- FastAPI

**AI / Computer Vision**
- OpenCV
- ONNX Runtime
- NumPy
- Face detection model
- Face embedding / recognition model

**Database**
- PostgreSQL

**Caching / Temporary State**
- Redis

**Object Storage**
- S3-compatible storage such as MinIO or AWS S3, introduced when image/document storage becomes necessary

---

# 1. Product Direction

The system should evolve in stages instead of trying to build full KYC from the beginning.

| Version | Product Stage | Main Capability |
|---|---|---|
| V1 | Photo Face Verification | Compare profile image with uploaded image |
| V2 | Face Enrollment | Create and manage reusable face templates |
| V3 | Live Face Verification | Camera capture, liveness, and face verification |
| V4 | KYC Verification | ID document, live face, liveness, and decision engine |
| V5 | Identity Verification Platform | Expose verification capabilities as reusable APIs |

---

# 2. High-Level Architecture

```text
React + TypeScript + Vite
            |
            | HTTPS / REST
            v
        FastAPI
            |
    -------------------
    |        |        |
    v        v        v
 Face     Business   Session
 Engine    Logic     Management
    |                 |
    v                 v
ONNX/OpenCV         Redis
    |
    v
PostgreSQL

Later:
PostgreSQL + Redis + S3/MinIO + optional GPU workers
```

---

# 3. Version 1: Photo Face Verification MVP

## 3.1 Business Objective

The user already has a profile image.

The user uploads another image.

The system determines whether both images likely contain the same person.

This version is **face verification**, not KYC.

## 3.2 User Flow

```text
Profile Image
      |
      v
Face Detection
      |
      v
Quality Check
      |
      v
Face Alignment
      |
      v
Embedding Generation
      |
      +----------------------+
                             |
                             v
                    Similarity Comparison
                             ^
                             |
      +----------------------+
      |
Uploaded Image
      |
      v
Face Detection
      |
      v
Quality Check
      |
      v
Face Alignment
      |
      v
Embedding Generation
```

## 3.3 Possible Results

- `MATCH`
- `NO_MATCH`
- `NO_FACE`
- `MULTIPLE_FACES`
- `LOW_QUALITY`
- `PROCESSING_ERROR`

## 3.4 Frontend

### Pages / Components

```text
FaceVerifyPage
├── ReferenceImage
├── UploadImage
├── ImagePreview
├── VerifyButton
└── VerificationResult
```

### Frontend Responsibilities

- Show existing profile image
- Allow user to upload a comparison image
- Validate image type and size
- Show image preview
- Submit images using `multipart/form-data`
- Display similarity score
- Display verification result
- Handle retry and error states

## 3.5 Backend Structure

```text
backend/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── v1/
│   │       └── face.py
│   ├── services/
│   │   ├── detection.py
│   │   ├── alignment.py
│   │   ├── embedding.py
│   │   ├── comparison.py
│   │   └── quality.py
│   ├── repositories/
│   │   └── face_repository.py
│   ├── core/
│   │   ├── config.py
│   │   └── security.py
│   └── db/
│       └── postgres.py
├── models/
│   ├── detector.onnx
│   └── recognizer.onnx
└── requirements.txt
```

## 3.6 Face Processing Pipeline

```text
Image
  |
  v
Decode
  |
  v
Detect Face
  |
  v
Validate Quality
  |
  v
Align Face
  |
  v
Generate Embedding
  |
  v
Normalize Embedding
  |
  v
Compare Embeddings
  |
  v
Decision
```

## 3.7 API Endpoints

### Detect Face

```http
POST /api/v1/face/detect
```

### Compare Faces

```http
POST /api/v1/face/compare
```

Request:

```text
multipart/form-data

reference_image
candidate_image
```

Example response:

```json
{
  "success": true,
  "face_detected": true,
  "matched": true,
  "similarity": 0.8421,
  "threshold": 0.72,
  "decision": "MATCH"
}
```

## 3.8 PostgreSQL

### users

```text
id
external_id
name
profile_image_url
created_at
updated_at
```

### face_verifications

```text
id
user_id
verification_type
reference_source
similarity_score
threshold
decision
detector_version
recognition_model_version
processing_time_ms
created_at
```

Recommended values:

```text
verification_type = PHOTO_COMPARE
reference_source = PROFILE_IMAGE
```

## 3.9 Redis

Redis is optional in V1.

Use it only if required for:

- rate limiting
- request deduplication
- short-lived verification sessions
- temporary processing results

Do not use Redis as the permanent biometric database.

## 3.10 V1 Acceptance Criteria

1. System must reject an image with no detectable face.
2. System must reject an image containing multiple faces when only one is expected.
3. System must reject images below the quality threshold.
4. System must generate embeddings for valid faces.
5. System must compare reference and candidate embeddings.
6. System must return similarity score and threshold.
7. System must store verification history.
8. System must record the AI model version used for the decision.

---

# 4. Version 2: Face Enrollment

## 4.1 Business Objective

Create a reusable biometric identity for each user instead of processing the original profile image for every verification request.

## 4.2 Flow

```text
Profile / Approved Image
        |
        v
Face Detection
        |
        v
Quality Validation
        |
        v
Face Alignment
        |
        v
Embedding Generation
        |
        v
Encrypted Face Template
        |
        v
PostgreSQL
```

Future verification:

```text
Candidate Image
      |
      v
Embedding
      |
      v
Compare
      |
      v
Stored Face Template
```

## 4.3 Enrollment Status

```text
NOT_ENROLLED
PENDING
ENROLLED
VERIFIED
REVOKED
RE_ENROLLMENT_REQUIRED
```

## 4.4 Reference Sources

```text
PROFILE_IMAGE
HR_APPROVED_IMAGE
LIVE_ENROLLMENT
NID
PASSPORT
OTHER_VERIFIED_DOCUMENT
```

## 4.5 Frontend

Add an enrollment screen:

```text
/face/enrollment
```

Responsibilities:

- show enrollment status
- show reference image
- show quality result
- allow enrollment
- allow re-enrollment
- show model/version information for admins
- show enrollment date and status

## 4.6 APIs

```http
POST /api/v1/face/enroll
GET /api/v1/face/enrollment/{user_id}
DELETE /api/v1/face/enrollment/{user_id}
POST /api/v1/face/verify
```

## 4.7 Database

### face_subjects

```text
id
user_id
status
created_at
updated_at
```

### face_templates

```text
id
face_subject_id
embedding
embedding_dimension
model_name
model_version
reference_source
quality_score
active
created_at
revoked_at
```

Keep biometric data separate from the `users` table.

Recommended relationship:

```text
users
  |
  v
face_subjects
  |
  v
face_templates
```

This allows template versioning and re-enrollment without modifying the user record.

## 4.8 Redis

Redis remains optional but can be introduced for:

```text
face:template:{user_id}
```

Possible use:

```text
Verification Request
        |
        v
Redis Lookup
   |         |
 HIT       MISS
   |         |
   |         v
   |     PostgreSQL
   |         |
   +---------+
        |
        v
Face Template
```

Because face templates are sensitive, cache them only when performance measurements justify it.

---

# 5. Version 3: Live Face Verification

## 5.1 Business Objective

Verify that the person currently using the application is the enrolled user.

This version introduces:

- browser camera
- live face capture
- liveness / presentation attack detection
- session-based verification

## 5.2 Flow

```text
Live Camera
     |
     v
Face Detection
     |
     v
Quality Check
     |
     v
Liveness Check
     |
     v
Best Frame Selection
     |
     v
Embedding Generation
     |
     v
Compare With Enrolled Template
     |
     v
VERIFIED / REJECTED
```

## 5.3 Frontend

Use:

```typescript
navigator.mediaDevices.getUserMedia({
  video: true
})
```

### Frontend Responsibilities

- camera permission
- camera preview
- face positioning guidance
- frame capture
- quality feedback
- liveness instructions
- progress state
- retry handling
- verification result

Do not continuously upload full 30 FPS video.

Recommended strategy:

```text
Camera Preview: 30 FPS
Processing: selected frames or approximately 2-5 FPS
```

## 5.4 Backend Architecture

```text
React
  |
  v
FastAPI API
  |
  v
Verification Service
  |
  +-----------------------+
  |          |            |
  v          v            v
Detector   Liveness   Recognition
  |          |            |
  +----------+------------+
             |
             v
        Decision Engine
```

## 5.5 Redis

Redis becomes important in V3.

Use it for live verification sessions.

Example keys:

```text
face:session:{uuid}
face:rate:{user_id}
face:challenge:{uuid}
face:result:{uuid}
```

Example session object:

```json
{
  "user_id": 123,
  "status": "PROCESSING",
  "face_detected": true,
  "liveness": true
}
```

Use short TTL values such as 2 to 5 minutes.

## 5.6 APIs

### Create Session

```http
POST /api/v1/live-verification/session
```

Example response:

```json
{
  "session_id": "abc123"
}
```

### Submit Frame

```http
POST /api/v1/live-verification/{session_id}/frame
```

### Complete Verification

```http
POST /api/v1/live-verification/{session_id}/complete
```

Example response:

```json
{
  "verified": true,
  "face_similarity": 0.84,
  "liveness_score": 0.96,
  "decision": "VERIFIED"
}
```

## 5.7 Infrastructure

Start with:

```text
FastAPI + ONNX Runtime + CPU
```

Move later to:

```text
FastAPI
   |
   v
Inference Workers
   |
   v
GPU
```

when real usage proves GPU is required.

---

# 6. Version 4: KYC Verification

## 6.1 Business Objective

Establish that a live person matches a trusted identity document.

At this stage, the reference identity should come from a trusted source such as:

- NID
- Passport
- Driving Licence
- other supported official identity documents

## 6.2 KYC Flow

```text
                  User
                    |
             +------+------+
             |             |
             v             v
       ID Document     Live Camera
             |             |
             v             v
     Document Detection  Face Detection
             |             |
             v             v
            OCR         Liveness
             |             |
             v             v
     Extract ID Face    Live Face
             |             |
             +------+------+
                    |
                    v
             Face Comparison
                    |
                    v
             Decision Engine
                    |
         +----------+----------+
         |          |          |
         v          v          v
     VERIFIED    REVIEW     REJECTED
```

## 6.3 KYC Decision Inputs

KYC should combine:

```text
Document Validity
+
OCR Confidence
+
Document Face Extraction
+
Live Face Detection
+
Liveness Result
+
Face Similarity
+
Business Rules
```

KYC should not be treated as only a face match.

## 6.4 Frontend

Recommended wizard:

```text
Step 1: Personal Information
Step 2: Upload NID / Passport
Step 3: Document Processing
Step 4: Live Face Capture
Step 5: Liveness Verification
Step 6: Verification Result
```

Suggested React structure:

```text
KycPage
├── PersonalInfoStep
├── DocumentUploadStep
├── DocumentPreviewStep
├── CameraStep
├── LivenessStep
└── ResultStep
```

## 6.5 Backend Modules

```text
app/
├── face/
│   ├── detection/
│   ├── recognition/
│   ├── alignment/
│   └── quality/
├── liveness/
│   └── detection/
├── documents/
│   ├── detection/
│   ├── preprocessing/
│   ├── ocr/
│   └── validation/
├── kyc/
│   ├── workflow/
│   ├── rules/
│   └── decisions/
└── audit/
```

KYC should orchestrate the face, liveness, OCR, and document-validation components.

## 6.6 PostgreSQL

### kyc_sessions

```text
id
user_id
status
document_type
started_at
completed_at
expires_at
```

### kyc_documents

```text
id
kyc_session_id
document_type
document_number
name
date_of_birth
document_image_url
ocr_confidence
created_at
```

### kyc_verifications

```text
id
kyc_session_id
document_valid
face_similarity
face_threshold
liveness_score
liveness_passed
decision
manual_review_required
created_at
```

## 6.7 Object Storage

Do not store large images directly in PostgreSQL unless there is a specific reason.

Recommended:

```text
PostgreSQL
    |
    +--> metadata and object references

S3 / MinIO
    |
    +--> profile images
    +--> KYC documents
    +--> reference images
    +--> temporary evidence images
```

---

# 7. Version 5: Identity Verification Platform

## 7.1 Business Objective

Turn the internal system into a reusable identity verification platform that can serve multiple products.

Potential consumers:

```text
HR / Attendance
Fintech
Banking
E-commerce
Access Control
Visitor Management
Other SaaS Products
```

## 7.2 API Surface

```http
POST /api/v1/faces/detect
POST /api/v1/faces/compare
POST /api/v1/faces/enroll
POST /api/v1/faces/verify

POST /api/v1/liveness/session

POST /api/v1/kyc/session
POST /api/v1/kyc/document
POST /api/v1/kyc/verify

GET /api/v1/kyc/{id}
```

Potential later additions:

- API keys
- tenant management
- webhooks
- SDKs
- quotas
- rate limits
- usage metering
- billing
- audit exports

---

# 8. Database Responsibility

## PostgreSQL

Use PostgreSQL for persistent business data:

- users
- face subjects
- enrollment metadata
- face templates
- verification history
- KYC sessions
- KYC results
- audit logs
- model versions
- threshold configurations
- tenant data

## Redis

Use Redis for temporary or high-speed data:

- verification sessions
- rate limiting
- challenge tokens
- short-lived workflow state
- temporary results
- distributed locks
- frequently accessed configuration
- job state

Redis should not be treated as the primary permanent database for biometric records.

## Object Storage

Use S3 / MinIO for:

- profile images
- ID documents
- enrollment images
- temporary captures
- KYC evidence

---

# 9. Final Target Architecture

```text
                     React
               TypeScript + Vite
                       |
                       | HTTPS
                       v
                +--------------+
                |   FastAPI    |
                | API Layer    |
                +------+-------+
                       |
        +--------------+---------------+
        |              |               |
        v              v               v
   Face Service    KYC Service    Authentication
        |              |
        |        +-----+------+
        |        |            |
        v        v            v
 Detection     OCR      Document Rules
 Alignment
 Embedding
 Liveness
        |
        v
   ONNX Runtime
        |
   +----+----+
   |         |
   v         v
  CPU       GPU
   |
   +---------------------------+
   |             |             |
   v             v             v
PostgreSQL     Redis       S3 / MinIO
```

---

# 10. Version-by-Version Technology Matrix

| Component | V1 | V2 | V3 | V4 | V5 |
|---|---:|---:|---:|---:|---:|
| React | Yes | Yes | Yes | Yes | Yes |
| TypeScript | Yes | Yes | Yes | Yes | Yes |
| Vite | Yes | Yes | Yes | Yes | Yes |
| FastAPI | Yes | Yes | Yes | Yes | Yes |
| OpenCV | Yes | Yes | Yes | Yes | Yes |
| ONNX Runtime | Yes | Yes | Yes | Yes | Yes |
| PostgreSQL | Yes | Yes | Yes | Yes | Yes |
| Redis | Optional | Optional | Yes | Yes | Yes |
| Object Storage | Optional | Recommended | Yes | Yes | Yes |
| Face Detection | Yes | Yes | Yes | Yes | Yes |
| Face Embedding | Yes | Yes | Yes | Yes | Yes |
| Face Enrollment | No | Yes | Yes | Yes | Yes |
| Camera Capture | No | No | Yes | Yes | Yes |
| Liveness | No | No | Yes | Yes | Yes |
| OCR | No | No | No | Yes | Yes |
| Document Verification | No | No | No | Yes | Yes |
| GPU | No | No | Optional | Optional / Scale | Scale dependent |

---

# 11. Business Roadmap

## V1: Face Match MVP

**Business Question**

Can the system reliably determine whether two supplied photographs contain the same person?

**Deliverables**

- image upload
- face detection
- image quality validation
- face alignment
- embedding generation
- face comparison
- similarity score
- configurable threshold
- verification history
- model version tracking

---

## V2: Face Identity Enrollment

**Business Question**

Can the platform create and manage a reusable biometric identity for a user?

**Deliverables**

- face enrollment
- biometric template
- enrollment status
- template versioning
- re-enrollment
- admin management
- verification history
- reference-source tracking

---

## V3: Live Identity Verification

**Business Question**

Can the platform verify that the person currently using the system is the enrolled user?

**Deliverables**

- camera capture
- live face detection
- image quality feedback
- liveness / anti-spoofing
- enrolled-face verification
- Redis session management
- retry rules
- audit trail

---

## V4: KYC

**Business Question**

Can the platform establish that a live person matches a trusted identity document?

**Deliverables**

- NID/passport upload
- document image processing
- OCR
- document face extraction
- live face capture
- liveness
- face comparison
- KYC decision engine
- manual review
- audit trail

---

## V5: Identity Verification Platform

**Business Question**

Can face verification and KYC be exposed as a reusable service to multiple products or customers?

**Deliverables**

- public/private API
- API authentication
- tenant isolation
- rate limiting
- usage tracking
- webhooks
- SDKs
- enterprise integrations
- monitoring and audit exports

---

# 12. Security Requirements

Biometric data should be treated as highly sensitive.

Minimum controls:

- HTTPS/TLS everywhere
- encryption at rest
- encrypted face templates
- strict access control
- tenant isolation
- audit logging
- retention policy
- deletion / revocation process
- signed object-storage URLs
- no public biometric URLs
- rate limiting
- request-size limits
- model/version tracking
- restricted admin access

Do not return face embeddings through normal user/profile APIs.

---

# 13. Core Engineering Principles

## Keep AI decisions separate from business decisions

The face engine should return:

```json
{
  "similarity": 0.84,
  "threshold": 0.72,
  "liveness_score": 0.96
}
```

The business layer should decide:

```text
MATCH
NO_MATCH
RETRY
REVIEW
REJECT
```

This keeps model behavior and business policy separate.

## Version Every Model

Every verification record should include:

```text
detector_name
detector_version
recognition_model_name
recognition_model_version
liveness_model_version
threshold_version
```

This is necessary when models are upgraded.

## Avoid Premature Infrastructure Complexity

For V1, start with:

```text
React + TypeScript + Vite
            |
            v
         FastAPI
            |
       +----+----+
       |         |
       v         v
    OpenCV     ONNX
       |         |
       +----+----+
            |
            v
       PostgreSQL
```

Do not introduce:

- Redis Cluster
- Kubernetes
- GPU cluster
- message queues
- complex microservices
- OCR
- KYC workflow
- advanced liveness

until the product requires them.

---

# 14. Recommended Initial Build Order

## V1 Development Sequence

1. Create FastAPI project structure.
2. Integrate face detector.
3. Integrate face embedding model.
4. Implement face alignment.
5. Implement image-quality validation.
6. Implement similarity comparison.
7. Define and calibrate initial threshold.
8. Create `/face/detect` API.
9. Create `/face/compare` API.
10. Create PostgreSQL schema.
11. Store verification history.
12. Build React upload interface.
13. Show similarity and result.
14. Test with a controlled image dataset.
15. Measure false matches and false rejections.
16. Adjust thresholds based on real data.
17. Prepare V2 enrollment design.

---

# 15. Recommended Starting Point

The first production milestone should be:

```text
React + TypeScript + Vite
            |
            v
         FastAPI
            |
            v
 Face Detection + Alignment
            |
            v
    Embedding Generation
            |
            v
    Similarity Comparison
            |
            v
       PostgreSQL
```

V1 should prove that the core pipeline works reliably:

```text
Image
  ->
Face Detection
  ->
Quality Validation
  ->
Alignment
  ->
Embedding
  ->
Comparison
  ->
Decision
```

Once this pipeline performs reliably with real users, proceed to face enrollment, then live verification, then KYC.

This approach allows every version to build on the previous one without replacing the core system.
