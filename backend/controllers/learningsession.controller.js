const LearningSession = require("../models/learningsession.model");

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

// Finish the learning session and calculate total and average time
exports.finishLearningSession = async (req, res) => {
  try {
    const sessionId = req.params.id;
    // Extract additional details from the request body
    const { finishedSession, currentMood, parentSatisfaction, engagementLevel } = req.body;
    const session = await LearningSession.findById(sessionId);
    if (!session) {
      return res.status(404).json({ message: "Learning session not found" });
    }
    if (finishedSession !== true) {
      return res.status(400).json({
        message: "finishedSession flag must be true to complete the session",
      });
    }

    // Mark the session as finished
    session.finishedSession = true;

    // Calculate total takenTime and average takenTime from all instructions
    const totalTime = session.InstructinRecords.reduce(
      (acc, record) => acc + record.takenTime,
      0
    );
    const averageTime =
      session.InstructinRecords.length > 0
        ? totalTime / session.InstructinRecords.length
        : 0;

    // Calculate completed tasks (count of instructions)
    const completedTasks = session.InstructinRecords.length;

    // Count correct answers on the first attempt (assuming isCorrect indicates that)
    const correctInFirstAttempt = session.InstructinRecords.filter(
      (record) => record.isCorrect === true
    ).length;

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
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
