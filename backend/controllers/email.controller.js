const LearningSession = require("../models/learningsession.model");
const Email = require("../models/email.model");
const nodemailer = require("nodemailer");
const PDFDocument = require("pdfkit");
const fs = require("fs");
const path = require("path");
const axios = require("axios");

// Helper function to generate the PDF report with all details and charts
async function generatePDF(
  session,
  takenTimeChartBuffer,
  isCorrectChartBuffer
) {
  return new Promise((resolve, reject) => {
    const doc = new PDFDocument({ margin: 50 });

    // Font registration (keep your existing font setup)
    const sinhalaFontPath = path.join(
      __dirname,
      "../Noto_Sans_Sinhala/NotoSansSinhala-VariableFont_wdth,wght.ttf"
    );
    console.log(sinhalaFontPath)

    if (fs.existsSync(sinhalaFontPath)) {
      doc.registerFont("Sinhala", sinhalaFontPath);
      doc.font("Sinhala");
    } else {
      console.error(
        "Sinhala font not found at " + sinhalaFontPath + ". Using default font."
      );
      doc.font("Helvetica");
    }

    // Ensure the reports directory exists
    const reportsDir = path.join(__dirname, "../reports");
    if (!fs.existsSync(reportsDir)) {
      fs.mkdirSync(reportsDir, { recursive: true });
    }

    // Define output file path using session _id
    const outputPath = path.join(reportsDir, `${session._id}.pdf`);
    const stream = fs.createWriteStream(outputPath);
    doc.pipe(stream);

    // --- Title & Header ---
    doc
      .fontSize(22)
      .fillColor("blue")
      .text("Final Progress Report", { align: "center" });
    doc.moveDown(0.5);

    // Draw a horizontal line
    doc
      .moveTo(50, doc.y)
      .lineTo(550, doc.y)
      .strokeColor("gray")
      .lineWidth(2)
      .stroke();
    doc.moveDown(1);

    // --- User Details Section ---
    doc
      .fontSize(16)
      .fillColor("blue")
      .text("User Details", { underline: true });
    doc.moveDown();
    doc
      .fontSize(12)
      .fillColor("gray")
      .text(`Full Name: ${session.user.fullName}`)
      .text(`Email: ${session.user.email}`)
      .text(`Age: ${session.user.age}`)
      .text(`Gender: ${session.user.gender}`);
    doc.moveDown();

    // --- Session Details Section ---
    doc
      .fontSize(16)
      .fillColor("blue")
      .text("Session Details", { underline: true });
    doc.moveDown();
    doc
      .fontSize(12)
      .fillColor("gray")
      .text(`Selected Location: ${session.selectedLocation}`)
      .text(`Detected Objects: ${session.detectedObjects}`)
      .text(`Parent Satisfaction: ${session.parentSatisfaction}`)
      .text(`Engagement Level: ${session.engagementLevel}`)
      .text(`Completed Tasks: ${session.completedTasks}`)
      .text(`Correct in First Attempt: ${session.correctInFirstAttempt}`)
      .text(`Prediction: ${session.prediction}`);
    doc.moveDown();

    // --- Emotion Analysis Section ---
    doc
      .fontSize(16)
      .fillColor("blue")
      .text("Emotion Analysis", { underline: true });
    doc.moveDown();

    if (session.emotionSnapshots.length > 0) {
      // Sort snapshots by type (Initial, Middle, Final)
      const sortedSnapshots = session.emotionSnapshots.sort((a, b) => {
        const order = { Initial: 0, Middle: 1, Final: 2 };
        return order[a.emotion_type] - order[b.emotion_type];
      });

      sortedSnapshots.forEach((snapshot) => {
        doc
          .fontSize(14)
          .fillColor("darkblue")
          .text(`${snapshot.emotion_type} State:`);

        doc
          .fontSize(12)
          .fillColor("gray")
          .text(`Dominant Emotion: ${snapshot.dominant_emotion}`)
          .text(`Timestamp: ${snapshot.timestamp.toLocaleString()}`);

        // Emotion percentages table
        doc.moveDown(0.5);
        doc.font("Helvetica-Bold").text("Emotion Distribution:");
        doc.font("Helvetica");

        // Create a table for emotion percentages
        const startX = 50;
        const startY = doc.y;
        const rowHeight = 20;
        const colWidth = 100;

        // Table headers
        doc.font("Helvetica-Bold");
        doc.text("Emotion", startX, startY);
        doc.text("Percentage", startX + colWidth, startY);
        doc.font("Helvetica");

        // Table rows
        Object.entries(snapshot.emotion_percentages).forEach(
          ([emotion, percentage], i) => {
            const y = startY + (i + 1) * rowHeight;
            doc.text(emotion, startX, y);
            doc.text(`${percentage.toFixed(2)}%`, startX + colWidth, y);
          }
        );

        doc.moveDown(
          (Object.keys(snapshot.emotion_percentages).length + 1) * 0.5
        );
      });
    } else {
      doc.text("No emotion data available for this session");
    }
    doc.moveDown();

    // --- Instruction Records Section ---
    doc
      .fontSize(16)
      .fillColor("blue")
      .text("Instruction Records", { underline: true });
    doc.moveDown();
    session.InstructinRecords.forEach((record, index) => {
      doc
        .fontSize(12)
        .fillColor("gray")
        .text(`Instruction ${index + 1}:`)
        .text(`  ID: ${record.instructioId}`)
        .text(`  Question: ${record.quesition}`)
        .text(`  Correct Answer: ${record.correctAnswer}`)
        .text(`  Wrong Answer: ${record.wrongAnswer}`)
        .text(`  Selected Answer: ${record.selectedAnswer}`)
        .text(`  Is Correct: ${record.isCorrect}`)
        .text(`  Taken Time: ${record.takenTime} seconds`)
        .moveDown();
    });

    // --- Charts Section ---
    // Add a new page for the first chart
    doc.addPage();
    doc
      .fontSize(16)
      .fillColor("blue")
      .text("Quiz Taken Time Chart", { align: "center" });
    doc.moveDown(1);
    doc.image(takenTimeChartBuffer, { fit: [500, 300], align: "center" });

    // Add a new page for the second chart
    doc.addPage();
    doc
      .fontSize(16)
      .fillColor("blue")
      .text("Quiz Correctness Chart", { align: "center" });
    doc.moveDown(1);
    doc.image(isCorrectChartBuffer, { fit: [500, 300], align: "center" });

    doc.end();
    stream.on("finish", () => resolve(outputPath));
    stream.on("error", (err) => reject(err));
  });
}

exports.sendSessionEmail = async (req, res) => {
  try {
    const { sessionId, email } = req.body;
    if (!sessionId || !email) {
      return res
        .status(400)
        .json({ message: "sessionId and email are required" });
    }

    // Fetch the learning session with populated user details
    const session = await LearningSession.findById(sessionId).populate(
      "user",
      "-password"
    );
    if (!session) {
      return res.status(404).json({ message: "Learning session not found" });
    }

    // --- Generate Chart Configurations using QuickChart ---
    // Chart 1: Taken Time per Instruction (Line Chart)
    const labels = session.InstructinRecords.map(
      (record) => record.instructioId
    );
    const takenTimes = session.InstructinRecords.map(
      (record) => record.takenTime
    );
    const takenTimeChartConfig = {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Taken Time (seconds)",
            data: takenTimes,
            fill: false,
            borderColor: "blue",
          },
        ],
      },
      options: { title: { display: true, text: "Quiz Taken Time" } },
    };
    const takenTimeChartUrl = `https://quickchart.io/chart?c=${encodeURIComponent(
      JSON.stringify(takenTimeChartConfig)
    )}`;

    // Chart 2: Correctness per Instruction (Bar Chart)
    const correctnessLabels = session.InstructinRecords.map(
      (record) => record.instructioId
    );
    const correctnessData = session.InstructinRecords.map((record) =>
      record.isCorrect ? 1 : 0
    );
    const isCorrectChartConfig = {
      type: "bar",
      data: {
        labels: correctnessLabels,
        datasets: [
          {
            label: "Correct (1) / Incorrect (0)",
            data: correctnessData,
            backgroundColor: correctnessData.map((val) =>
              val === 1 ? "green" : "red"
            ),
          },
        ],
      },
      options: {
        title: { display: true, text: "Quiz Correctness" },
        scales: { yAxes: [{ ticks: { min: 0, max: 1, stepSize: 1 } }] },
      },
    };
    const isCorrectChartUrl = `https://quickchart.io/chart?c=${encodeURIComponent(
      JSON.stringify(isCorrectChartConfig)
    )}`;

    // --- Download Chart Images as Buffers ---
    const takenTimeChartResponse = await axios.get(takenTimeChartUrl, {
      responseType: "arraybuffer",
    });
    const isCorrectChartResponse = await axios.get(isCorrectChartUrl, {
      responseType: "arraybuffer",
    });
    const takenTimeChartBuffer = takenTimeChartResponse.data;
    const isCorrectChartBuffer = isCorrectChartResponse.data;

    // --- Generate the PDF Report with All Details and Charts ---
    const pdfPath = await generatePDF(
      session,
      takenTimeChartBuffer,
      isCorrectChartBuffer
    );

    // --- Create Nodemailer Transporter using Gmail Configuration ---
    const transporter = nodemailer.createTransport({
      service: "Gmail",
      auth: {
        user: process.env.EMAIL_ACCOUNT,
        pass: process.env.EMAIL_PASSWORD,
      },
    });

    // Configure mail options with PDF attachment
    const mailOptions = {
      from: process.env.EMAIL_ACCOUNT,
      to: email,
      subject: "Final Progress Report",
      text: "Please find attached your final progress report with detailed session information.",
      attachments: [
        {
          filename: `${sessionId}.pdf`,
          path: pdfPath,
        },
      ],
    };

    // --- Send Email ---
    const info = await transporter.sendMail(mailOptions);

    // Optionally, log the sent email
    const emailLog = new Email({
      session: session._id,
      to: email,
      subject: mailOptions.subject,
      body: mailOptions.text,
    });
    await emailLog.save();

    res.status(200).json({ message: "Email sent successfully", info });
  } catch (error) {
    console.error("Error sending email:", error);
    res.status(500).json({ error: error.message });
  }
};
