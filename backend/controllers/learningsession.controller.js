const LearningSession = require("../models/learningsession.model");
const User = require("../models/user.model");
const axios = require("axios");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const FormData = require("form-data");

// Set up multer storage configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = "uploads/";
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir);
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(
      null,
      file.fieldname + "-" + uniqueSuffix + path.extname(file.originalname)
    );
  },
});

const upload = multer({ storage: storage });

const videoUpload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    if (file.fieldname === "video" && file.mimetype.startsWith("video/")) {
      cb(null, true);
    } else {
      cb(
        new Error("Only video files are allowed for emotion detection!"),
        false
      );
    }
  },
  limits: { fileSize: 100 * 1024 * 1024 }, // 100MB max
}).single("video");

// Handle file upload and object detection
exports.uploadPhotoAndDetect = [
  upload.single("photo"),
  async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ message: "No file uploaded" });
      }

      const filePath = req.file.path;
      const formData = new FormData();
      formData.append(
        "file",
        fs.createReadStream(filePath),
        req.file.originalname
      );

      console.log("Sending form data:", formData);

      // Send image to external API for object detection
      const response = await axios.post(
        `${process.env.FLASH_BACKEND}/detect_objects/`,
        formData,
        {
          headers: {
            ...formData.getHeaders(),
            accept: "application/json",
          },
        }
      );

      // Get the detected objects from the response
      const detectedObjects = response.data.detected_objects;

      // Filter detected objects with confidence > 0.5
      const filteredObjects = detectedObjects.filter(
        (obj) => obj.confidence > 0.5
      );

      // Use a Set to keep track of the unique class names
      const uniqueClasses = new Set();

      // Filter out duplicates by class (only keep the first occurrence of each class)
      const uniqueObjects = filteredObjects
        .filter((obj) => {
          if (!uniqueClasses.has(obj.class)) {
            uniqueClasses.add(obj.class);
            return true;
          }
          return false;
        })
        // Remove bbox from the final response
        .map((obj) => {
          const { bbox, ...rest } = obj;
          return rest;
        });

      // Return the final unique objects with confidence > 0.5 and only one object per class
      res.status(200).json({ detectedObjects: uniqueObjects });
    } catch (error) {
      console.error("Error during external API request:", error);
      res.status(500).json({ error: error.message });
    }
  },
];

// Create a new learning session (user has logged in and selected a place)
exports.createLearningSession = async (req, res) => {
  try {
    const userId = req.user.id;
    const { selectedLocation, detectedObjects } = req.body;
    const newSession = new LearningSession({
      user: userId,
      selectedLocation,
      detectedObjects,
    });
    await newSession.save();
    res.status(201).json(newSession);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// Get Emotions
exports.addEmotionSnapshot = [
  (req, res, next) => {
    videoUpload(req, res, (err) => {
      if (err) {
        return res.status(400).json({ message: err.message });
      }
      next();
    });
  },
  async (req, res) => {
    try {
      const sessionId = req.params.id;
      const { emotionType } = req.body; // Should be "Initial", "Middle", or "Final"

      if (!req.file) {
        return res.status(400).json({ message: "No video file uploaded" });
      }

      // Validate emotion type
      const validEmotionTypes = ["Initial", "Middle", "Final"];
      if (!validEmotionTypes.includes(emotionType)) {
        // Clean up uploaded file if validation fails
        fs.unlink(req.file.path, (err) => {
          if (err) console.error("Error deleting temp file:", err);
        });
        return res.status(400).json({
          message:
            "Invalid emotion type. Must be one of: Initial, Middle, Final",
        });
      }

      const filePath = req.file.path;
      const formData = new FormData();
      formData.append(
        "file",
        fs.createReadStream(filePath),
        req.file.originalname
      );

      // Send video to emotion detection API
      const emotionResponse = await axios.post(
        `${process.env.FLASH_BACKEND}/detect_emotion_video/`,
        formData,
        {
          headers: {
            ...formData.getHeaders(),
            accept: "application/json",
          },
        }
      );

      // Create emotion snapshot object
      const emotionSnapshot = {
        emotion_type: emotionType,
        emotion_percentages: emotionResponse.data.emotion_percentages,
        dominant_emotion: emotionResponse.data.dominant_emotion,
        timestamp: new Date(),
      };

      // Update the learning session
      const session = await LearningSession.findByIdAndUpdate(
        sessionId,
        { $push: { emotionSnapshots: emotionSnapshot } },
        { new: true }
      );

      if (!session) {
        // Clean up uploaded file if session not found
        fs.unlink(filePath, (err) => {
          if (err) console.error("Error deleting temp file:", err);
        });
        return res.status(404).json({ message: "Learning session not found" });
      }

      // Clean up uploaded file after successful processing
      fs.unlink(filePath, (err) => {
        if (err) console.error("Error deleting temp file:", err);
      });

      res.status(200).json({
        message: "Emotion snapshot added successfully",
        session,
        emotionSnapshot,
      });
    } catch (error) {
      console.error("Error processing emotion snapshot:", error);
      // Clean up uploaded file if error occurs
      if (req.file) {
        fs.unlink(req.file.path, (err) => {
          if (err) console.error("Error deleting temp file:", err);
        });
      }
      res.status(500).json({
        error: error.message,
        details: error.response?.data || "No additional error details",
      });
    }
  },
];

// Append a new learning instruction to the session (using PUT method)
exports.addLearningInstruction = async (req, res) => {
  try {
    const sessionId = req.params.id;
    // Expecting the instruction object in the request body.
    const instruction = req.body;
    const session = await LearningSession.findById(sessionId);
    if (!session) {
      return res.status(404).json({ message: "Learning session not found" });
    }
    // Push new instruction to the array (without replacing existing ones)
    session.InstructinRecords.push(instruction);
    await session.save();
    res.status(200).json(session);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// Finish the learning session and get the prediction
exports.finishLearningSession = async (req, res) => {
  try {
    const sessionId = req.params.id;
    const { finishedSession, parentSatisfaction, engagementLevel } = req.body;

    // Fetch the learning session
    let session = await LearningSession.findById(sessionId);
    if (!session) {
      return res.status(404).json({ message: "Learning session not found" });
    }
    if (finishedSession !== true) {
      return res.status(400).json({
        message: "finishedSession flag must be true to complete the session",
      });
    }

    // Get the last emotion snapshot's dominant emotion as currentMood
    let currentMood = null;
    if (session.emotionSnapshots.length > 0) {
      // Sort snapshots by timestamp in descending order to get the most recent
      const sortedSnapshots = [...session.emotionSnapshots].sort(
        (a, b) => b.timestamp - a.timestamp
      );
      currentMood = sortedSnapshots[0].dominant_emotion;
    }

    // Update session details
    session.finishedSession = true;
    session.currentMood = currentMood; // Set from emotion snapshot
    session.parentSatisfaction = parentSatisfaction;
    session.engagementLevel = engagementLevel;

    // Calculate total takenTime and average takenTime
    const totalTime = session.InstructinRecords.reduce(
      (acc, record) => acc + record.takenTime,
      0
    );
    const averageTime =
      session.InstructinRecords.length > 0
        ? totalTime / session.InstructinRecords.length
        : 0;

    // Calculate completed tasks and count of correct answers on first attempt
    const completedTasks = session.InstructinRecords.length;
    const correctInFirstAttempt = session.InstructinRecords.filter(
      (record) => record.isCorrect === true
    ).length;

    // Update the session fields for tasks and correct attempts
    session.completedTasks = completedTasks;
    session.correctInFirstAttempt = correctInFirstAttempt;

    // Get user details (Age and Gender) from the User model
    const user = await User.findById(session.user);
    if (!user) {
      return res.status(404).json({ message: "User not found" });
    }

    // Construct the payload for the prediction API
    const predictPayload = {
      Age: user.age || 0,
      Gender: user.gender || "",
      Current_Mood: currentMood || "Neutral", 
      Parent_Satisfaction: parentSatisfaction,
      Engagement_Level: engagementLevel,
      Completed_Tasks: completedTasks,
      Time_Spent: totalTime,
      Correct_in_First_Attempt: correctInFirstAttempt,
    };

    console.log("Prediction Payload: ", predictPayload);

    // Call the prediction API
    let prediction, suggestions;
    try {
      const predictResponse = await axios.post(
        `${process.env.FLASH_BACKEND}/predict`,
        predictPayload
      );
      prediction = predictResponse.data.prediction;
      suggestions = predictResponse.data.suggestions;
      session.prediction = prediction;
      session.suggestions = suggestions;
    } catch (axiosError) {
      console.error(
        "Prediction API error: ",
        axiosError.response ? axiosError.response.data : axiosError.message
      );
      return res.status(500).json({
        error: "Prediction API call failed",
        details: axiosError.response
          ? axiosError.response.data
          : axiosError.message,
      });
    }

    // Save the session with updated values, prediction, and suggestions
    await session.save();

    res.status(200).json({
      session,
      totalTime,
      averageTime,
      currentMood,
      parentSatisfaction,
      engagementLevel,
      completedTasks,
      correctInFirstAttempt,
      prediction,
      suggestions,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// Get User All Learning Sessions
exports.getUserLearningSessions = async (req, res) => {
  try {
    const userId = req.user.id;

    const sessions = await LearningSession.find({ user: userId }).sort({ createdAt: -1 });

    return res.status(200).json({
      success: true,
      message: "User's learning sessions retrieved successfully",
      data: sessions,
    });
  } catch (error) {
    return res.status(500).json({
      success: false,
      message: "Server error",
      data: { error: error.message },
    });
  }
};


// Get a single learning session by its ID
exports.getLearningSession = async (req, res) => {
  try {
    const sessionId = req.params.id;
    const session = await LearningSession.findById(sessionId).populate(
      "user",
      "-password"
    );
    if (!session) {
      return res.status(404).json({ message: "Learning session not found" });
    }
    res.status(200).json(session);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
