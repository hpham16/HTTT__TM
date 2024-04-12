import express from 'express';
import bodyParser from 'body-parser';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import cors from 'cors';
import postRoutes from './routes/posts.js';
import userRoutes from './routes/user.js';

const app = express();
dotenv.config();


const PORT = process.env.PORT || 3000;


app.use(cors());


app.use(express.json({  limit: '30mb', extended: true }));
app.use(express.urlencoded({  limit: '30mb', extended: true }));


app.use('/posts', postRoutes);
app.use('/user', userRoutes);

app.get('/', (req, res) => {
    res.send('Hello');
});

//MongoDb

mongoose.connect("mongodb+srv://20521372:hung123@cluster0.z2ou3dg.mongodb.net/?retryWrites=true&w=majority&appName=cluster0", { useNewUrlParser: true, useUnifiedTopology: true })
    .then(() => app.listen(PORT, () => console.log(`Server running on port: ${PORT}`)))
    .catch((error) => console.log(error.message));
