import React, { useState, useEffect } from 'react';
import { TextField, Button, Typography, Paper } from '@material-ui/core';
import FileBase from 'react-file-base64';
import { useDispatch, useSelector } from 'react-redux';
import { useHistory } from 'react-router-dom';
import useStyles from './styles.js';
import { createPost, updatePost } from  '../../actions/posts.js';
import ChipInput from 'material-ui-chip-input';




const Form = ({ currentId, setCurrentId }) => {
    const [postData, setPostData] = useState({
        title: "", message: "", tags: [], selectedField: "",
    });
    const post = useSelector((state) => currentId ? state.posts.posts.find((p) => p._id === currentId) : null)
    const classes = useStyles();
    const dispatch = useDispatch();
    const user = JSON.parse(localStorage.getItem('profile'));
    const history = useHistory();

    useEffect(() => {
        if(post) setPostData(post);
    }, [post])

    const handleSubmit = (e) => {
        e.preventDefault();

        if(currentId) {
            dispatch(updatePost(currentId, {...postData, name: user?.result?.name}));
        } else {
            dispatch(createPost({...postData, name: user?.result?.name}, history));
        }

        clear();

    }

    const clear = () => {
        setCurrentId(null);
        setPostData({title: "", message: "", tags: [], selectedField: "",});
    }

    const handleAddTag = (tag) => {
        setPostData({...postData, tags: [...postData.tags, tag]});
    }

    const handleDeleteTag = (tagToDelete) => {
        setPostData({ ...postData, tags: postData.tags.filter((tag) => tag !== tagToDelete) });
      };

    if(!(user?.result?.name)) {
        return (
            <Paper className={classes.paper}>
                <Typography variant="h6" align="center">
                    Please Sign In to create your own memories and like other's memories
                </Typography>
            </Paper>
        )
    }

    return (
        <Paper className={classes.paper} elevation={6}>
            <form autoComplete="off" noValidate className={`${classes.root} ${classes.form}`} onSubmit={handleSubmit}>
                <Typography variant="h6">{currentId ? 'Editing' : 'Creating'} a Memory</Typography>
                <TextField name="title" id="outlined-basic" label="Title" variant="outlined" fullWidth value={postData.title} onChange={(e) => setPostData({ ...postData, title: e.target.value })}/>
                <TextField
                    name="message"
                    id="outlined-basic"
                    label="Message"
                    multiline
                    rows={4}                 
                    variant="outlined"
                    fullWidth
                    value={postData.message}
                    inputProps={{
                        maxlength: 300
                    }}
                    helperText={`${postData.message.length}/${300}`}
                    onChange={(e) => setPostData({...postData, message: e.target.value })}
                />
                <div style={{ padding: '5px 0', width: '96%' }}>
                    <ChipInput
                        name="tags"
                        variant="outlined"
                        label="Tags"
                        fullWidth
                        value={postData.tags}
                        onAdd={(tag) => handleAddTag(tag)}
                        onDelete={(tag) => handleDeleteTag(tag)}
                    />
                </div>
               
                <div className={classes.fileInput}>
                    <FileBase type="file" mutiple={false} onDone={({base64}) => setPostData({ ...postData, selectedFile: base64 })}/>
                </div>
                <Button className={classes.buttonSubmit} variant="contained" color="primary" size="large" type="submit" fullWidth>Submit</Button>
                <Button variant="contained" color="secondary" size="small" onClick={clear} fullWidth>Clear</Button>
            </form>
        </Paper>
    );
}

export default Form;