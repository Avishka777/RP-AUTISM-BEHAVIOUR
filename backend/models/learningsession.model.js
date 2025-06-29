const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const emotionSnapshotSchema = new Schema(
  {
    emotion_type: {
      type: String,
      enum: ["Initial", "Middle", "Final"],
      required: true,
    },
    emotion_percentages: {
      Angry: { type: Number, default: 0 },
      Disgust: { type: Number, default: 0 },
      Fear: { type: Number, default: 0 },
      Happy: { type: Number, default: 0 },
      Sad: { type: Number, default: 0 },
      Surprise: { type: Number, default: 0 },
      Neutral: { type: Number, default: 0 },
    },
    dominant_emotion: {
      type: String,
      enum: ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral","Frustrated"],
      default: null,
    },
    timestamp: {
      type: Date,
      default: Date.now,
    },
  },
  { _id: false }
);

const learningInstructionSchema = new Schema(
  {
    instructioId: {
      type: Number,
      required: true,
    },
    quesition: {
      type: String,
      required: true,
    },
    correctAnswer: {
      type: String,
      required: true,
    },
    wrongAnswer: {
      type: String,
      required: true,
    },
    selectedAnswer: {
      type: String,
      required: true,
    },
    isCorrect: {
      type: Boolean,
      required: true,
    },
    takenTime: {
      type: Number,
      required: true,
    },
  },
  { _id: false }
);

const LearningSessionSchema = new Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    required: true,
  },
  selectedLocation: {
    type: String,
    default: null,
  },
  detectedObjects: {
    type: String,
    default: null,
  },
  InstructinRecords: {
    type: [learningInstructionSchema],
    default: [],
  },
  emotionSnapshots: {
    type: [emotionSnapshotSchema],
    default: [],
  },
  finishedSession: {
    type: Boolean,
    default: false,
  },
  currentMood: {
    type: String,
    default: null,
  },
  parentSatisfaction: {
    type: Number,
    default: null,
  },
  engagementLevel: {
    type: Number,
    default: null,
  },
  completedTasks: {
    type: Number,
    default: 0,
  },
  correctInFirstAttempt: {
    type: Number,
    default: 0,
  },
  prediction: {
    type: String,
    default: null,
  },
  suggestions: {
    type: [String],
    default: [],
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

module.exports = mongoose.model("LearningSession", LearningSessionSchema);
