from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    Float,
    DateTime,
    ForeignKey
)

from app.core.database import Base


class PredictionHistory(Base):

    __tablename__ = "prediction_history"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    age = Column(
        Integer,
        nullable=False
    )

    bp = Column(
        Integer,
        nullable=False
    )

    creatinine = Column(
        Float,
        nullable=False
    )

    prediction = Column(
        Integer,
        nullable=False
    )

    probability = Column(
        Float,
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )