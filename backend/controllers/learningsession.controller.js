const LearningSession = require("../models/learningsession.model");
const User = require("../models/user.model");
const axios = require("axios");

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
    const {
      finishedSession,
      currentMood,
      parentSatisfaction,
      engagementLevel,
    } = req.body;

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

    // Update session details
    session.finishedSession = true;
    session.currentMood = currentMood;
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
      Current_Mood: currentMood,
      Parent_Satisfaction: parentSatisfaction,
      Engagement_Level: engagementLevel,
      Completed_Tasks: completedTasks,
      Time_Spent: totalTime,
      Correct_in_First_Attempt: correctInFirstAttempt,
    };

    console.log("Prediction Payload: ", predictPayload);

    // Call the prediction API with
    let prediction;
    try {
      const predictResponse = await axios.post(
        `${process.env.FLASH_BACKEND}/predict`,
        predictPayload
      );
      prediction = predictResponse.data.prediction;
      session.prediction = prediction;
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

    // Save the session with updated values and prediction
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
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
