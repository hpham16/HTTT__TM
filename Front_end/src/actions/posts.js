import * as api from '../api';
import axios from 'axios';
import { FETCH_ALL, FETCH_POST, FETCH_BY_SEARCH, START_LOADING, END_LOADING, CREATE, UPDATE, DELETE, LIKE, COMMENT} from '../constants/actionTypes.js'
//Action Creators
export const getPost = (id) => async (dispatch) => {
    try {
        dispatch({ type: START_LOADING });
        const { data } = await api.fetchPost(id);

        dispatch({ type: FETCH_POST, payload: data  });
        dispatch({ type: END_LOADING });
    } catch (error) {
        console.log(error);
    }
}


export const getPosts = (page) => async (dispatch) => {
    try {
        dispatch({ type: START_LOADING });
        const { data } = await api.fetchPosts(page);
        dispatch({ type: FETCH_ALL, payload: data  });
        dispatch({ type: END_LOADING });
    } catch (error) {
        console.log(error);
    }
}

export const getPostsBySearch = (searchQuery) => async (dispatch) => {
    try {
        dispatch({ type: START_LOADING });
        const { data: { data } } = await api.fetchPostsBySearch(searchQuery);
        dispatch({type: FETCH_BY_SEARCH, payload: data });
        dispatch({ type: END_LOADING });

    } catch (error) {
        console.log(error);
    }
}


export const createPost = (post, history) => async (dispatch) => {
    try {
        dispatch({ type: START_LOADING });
        const { data } = await api.createPost(post);

        history.push(`/posts/${data._id}`);

        dispatch({ type: CREATE, payload: data })
    } catch (error) {
        console.log(error);
    }
}

export const updatePost = (id, post) => async (dispatch) => {
    try {
        const { data } = await api.updatePost(id, post);

        dispatch({ type: UPDATE, payload: data});
    } catch (error) {
        console.log(error);
    }
}

export const deletePost = (id) => async (dispatch) => {
    try {
      await api.deletePost(id);
  
      dispatch({ type: DELETE, payload: id });
    } catch (error) {
      console.log(error);
    }
};

export const likePost = (id) => async (dispatch) => {
    const user = JSON.parse(localStorage.getItem('profile'));
  
    try {
      const { data } = await api.likePost(id, user?.token);
  
      dispatch({ type: LIKE, payload: data });
    } catch (error) {
      console.log(error);
    }
};

// export const commentPost = (value, id) => async (dispatch) => {
//     try {
//         const { data } = await api.comment(value, id);

//         dispatch({ type: COMMENT, payload: data });

//         return data.comments;
//     } catch (error) {
//         console.log(error);
//     }
// }


export const commentPost = (value, id) => async (dispatch) => {
    try {
        // Gọi API để xác nhận cảm xúc của bình luận
        const response = await api.checkcmt(value)
        console.log(response.data);

        // Kiểm tra nếu cảm xúc là tiêu cực
        if (response.data === 1) {
            console.log("Bình luận tiêu cực và không được đăng lên.");
            return null; 
        }

        // Nếu không, gửi bình luận xuống backend và dispatch action
        const {data} = await api.comment(value, id);
        dispatch({ type: COMMENT, payload: data });

        // Trả về dữ liệu comments từ API
        return data.comments;
    } catch (error) {
        console.log(error);
    }
}