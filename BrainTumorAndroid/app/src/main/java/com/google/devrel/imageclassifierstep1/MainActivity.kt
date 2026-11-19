// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


package com.google.devrel.imageclassifierstep1

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private lateinit var interpreter: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val img: ImageView = findViewById(R.id.imageToLabel)
        val txtOutput: TextView = findViewById(R.id.txtOutput)
        val btn: Button = findViewById(R.id.btnTest)

        val modelBytes = assets.open("brain_tumor_model.tflite").use {
            it.readBytes()
        }

        val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size)
        modelBuffer.order(ByteOrder.nativeOrder())
        modelBuffer.put(modelBytes)
        modelBuffer.rewind()

        interpreter = Interpreter(modelBuffer)

        val availableFiles = assets.list("")?.toList() ?: emptyList()

        val imageFiles = listOf(
            "mri1.jpg",
            "mri2.jpg",
            "mri3.jpg",
            "mri4.jpg"
        ).filter { it in availableFiles }

        btn.text = "Classify MRI"

        if (imageFiles.isEmpty()) {
            txtOutput.text = "No MRI images found in assets."
            return
        }

        var imageIndex = 0

        btn.setOnClickListener {

            val fileName = imageFiles[imageIndex]
            val bitmap = assetsToBitmap(fileName)

            if (bitmap != null) {

                img.setImageBitmap(bitmap)

                val input = prepareImage(bitmap)
                val output = Array(1) { FloatArray(1) }

                interpreter.run(input, output)

                val score = output[0][0]

                val prediction = if (score >= 0.5f) {
                    "Tumor present"
                } else {
                    "No tumor"
                }

                txtOutput.text =
                    "Image: $fileName\n" +
                            "Prediction: $prediction\n" +
                            "Tumor prediction score: %.4f".format(score)

            } else {
                txtOutput.text = "Could not open $fileName"
            }

            imageIndex = (imageIndex + 1) % imageFiles.size
        }
    }

    private fun prepareImage(bitmap: Bitmap): ByteBuffer {

        val size = 128

        val resized = Bitmap.createBitmap(
            size,
            size,
            Bitmap.Config.ARGB_8888
        )

        val canvas = Canvas(resized)
        canvas.drawColor(Color.BLACK)

        val scale = minOf(
            size.toFloat() / bitmap.width,
            size.toFloat() / bitmap.height
        )

        val width = bitmap.width * scale
        val height = bitmap.height * scale

        val left = (size - width) / 2f
        val top = (size - height) / 2f

        val destination = RectF(
            left,
            top,
            left + width,
            top + height
        )

        val paint = Paint(Paint.FILTER_BITMAP_FLAG)

        canvas.drawBitmap(
            bitmap,
            null,
            destination,
            paint
        )

        val pixels = IntArray(size * size)

        resized.getPixels(
            pixels,
            0,
            size,
            0,
            0,
            size,
            size
        )

        val input = ByteBuffer.allocateDirect(
            4 * size * size
        )

        input.order(ByteOrder.nativeOrder())

        for (pixel in pixels) {

            val gray = (
                    0.299f * Color.red(pixel) +
                            0.587f * Color.green(pixel) +
                            0.114f * Color.blue(pixel)
                    ) / 255.0f

            input.putFloat(gray)
        }

        input.rewind()

        return input
    }

    override fun onDestroy() {
        if (::interpreter.isInitialized) {
            interpreter.close()
        }
        super.onDestroy()
    }
}

fun Context.assetsToBitmap(fileName: String): Bitmap? {
    return try {
        with(assets.open(fileName)) {
            BitmapFactory.decodeStream(this)
        }
    } catch (e: IOException) {
        null
    }
}
