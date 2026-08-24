"""Create V1 users and face verification audit tables."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260824_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DECISIONS = (
    "'MATCH', 'NO_MATCH', 'REVIEW', 'NO_FACE', 'MULTIPLE_FACES', 'LOW_QUALITY', 'PROCESSING_ERROR'"
)


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("external_id", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("profile_image_url", sa.String(2048), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("external_id", name="uq_users_external_id"),
    )
    op.create_index("ix_users_external_id", "users", ["external_id"])

    op.create_table(
        "face_verifications",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("verification_type", sa.String(32), nullable=False),
        sa.Column("reference_source", sa.String(32), nullable=False),
        sa.Column("similarity_score", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("threshold_version", sa.String(128), nullable=False),
        sa.Column("decision", sa.String(32), nullable=False),
        sa.Column("reason_code", sa.String(64), nullable=True),
        sa.Column("quality_issues", sa.JSON(), nullable=False),
        sa.Column("reference_status", sa.String(32), nullable=False),
        sa.Column("candidate_status", sa.String(32), nullable=False),
        sa.Column("reference_face_count", sa.Integer(), nullable=False),
        sa.Column("candidate_face_count", sa.Integer(), nullable=False),
        sa.Column("detector_version", sa.String(128), nullable=False),
        sa.Column("recognition_model_version", sa.String(128), nullable=False),
        sa.Column("processing_time_ms", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            f"decision IN ({DECISIONS})", name="ck_face_verifications_decision_valid"
        ),
        sa.CheckConstraint(
            "similarity_score IS NULL OR (similarity_score >= -1 AND similarity_score <= 1)",
            name="ck_face_verifications_similarity_range",
        ),
        sa.CheckConstraint(
            "threshold >= -1 AND threshold <= 1", name="ck_face_verifications_threshold_range"
        ),
        sa.CheckConstraint(
            "processing_time_ms >= 0", name="ck_face_verifications_processing_time_nonnegative"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="CASCADE",
            name="fk_face_verifications_user_id_users",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_face_verifications"),
    )
    op.create_index("ix_face_verifications_user_id", "face_verifications", ["user_id"])
    op.create_index("ix_face_verifications_decision", "face_verifications", ["decision"])
    op.create_index("ix_face_verifications_created_at", "face_verifications", ["created_at"])


def downgrade() -> None:
    op.drop_table("face_verifications")
    op.drop_table("users")
