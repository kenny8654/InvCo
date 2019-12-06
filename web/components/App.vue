<template lang='pug'> 
#app
  .ui.visible.sidebar.left.inverted.vertical.menu
    div.title
      img.ui.tiny.circular.centered.image#title(src='https://i.imgur.com/EIi4jhh.png')
      h1#title_text {{ title }}
    .ui.item(@click='show_home') Home
    .ui.item Admin
    .ui.item Dashboard
    .ui.item Statistic
  .ui.internally.celled.grid.pusher
    .row
      .four.wide.column
        h2 Number of recipes :{{recipe}}
      .four.wide.column
        h2 Number of image  : {{numberOfImages}}
      .four.wide.column
        h2 Empty
    .row
  div.pusher(v-if="page==='home'")
    .ui.internally.celled.grid
      .row 
        .thirteen.wide.column
          .ui.seven.stackable.cards
            .ui.link.card
              .image
                img(src='https://i.imgur.com/yNwH3jV.jpg')
            .ui.link.card
              .image
                img(src='https://i.imgur.com/yNwH3jV.jpg')
            .ui.link.card(v-for='im in imgList')
              .image
                img(v-bind:src="im.url")          
            .ui.link.card(@click ='show_upload')
              .image
                img(src="https://i.imgur.com/GWvodHn.png")
            
  div.pusher(v-else-if="page==='upload'")
    .ui.internally.celled.grid
      .four.wide.column
      div
        div.file-upload-content
          img.file-upload-image(src="#")
          div.image-title-wrap
            span.image-title Uploaded Image
        form(action='' method='post' enctype="multipart/form-data")
          div.image-upload-wrap
            input.file-upload-input(type='file' ref='file' @change='handleFileUpload()' accept='image/*' name='img')
            div.drag-text 
              h3 Drag and drop a file or select add Image
            
          div.file-upload
            button.file-upload-btn(type='button' @click='submitFile()' v-if="upload_status==='waiting'") Upload
          h3(v-if="upload_status==='success'") Success Uploaging
</template>


<script>
import axios from "axios";
import "semantic-ui-offline/semantic.min.css";
import "jquery";
import $ from "jquery";

export default {
  beforeDestory() {
    document.removeEventListener("keyup", this.key);
  },

  created() {
    axios.get("/getImages").then(response => {
      var img_list = JSON.parse(response.data.replace(/'/g, '"'));
      var count = 0;
      var tmp_img_list = []
      console.log(img_list.images);
      for(var img of img_list.images){
        tmp_img_list.push({'url':'http://yichen.ee.ncku.edu.tw:10122/media/img/' + img + "?t="});
        count++
      }
      this.imgList = tmp_img_list;
      this.numberOfImages = count;
    });
    document.addEventListener("keyup", this.key);
  },

  data() {
    return {
      title: "Inverse Cooking",
      recipe: 100,
      numberOfImages: 0,
      page: "home",
      file: "",
      uploadPercentage: 0,
      imgList: "",
      upload_status: "waiting"
    };
  },

  methods: {
    submitFile() {
      let formData = new FormData();
      formData.append("image", this.file); //required
      console.log(formData.get("image"));

      axios
        .post("/upload", formData, {
          headers: { "X-CSRFToken": this.getCookie("csrftoken") },
          onUploadProgress: function(progressEvent) {
            this.uploadPercentage = parseInt(
              Math.round((progressEvent.loaded * 100) / progressEvent.total)
            );
          }.bind(this)
        })
        .then(response => {
          console.log(response);
          if(response.status == '200'){
            this.upload_status = "success"
            console.log('success upload!')
          }
        });
    },
    handleFileUpload(event) {
      this.file = this.$refs.file.files[0];
      var name = this.file.name;
      if (this.file) {
        var reader = new FileReader();
        reader.onload = function(e) {
          $(".image-upload-wrap").hide();
          $(".file-upload-image").attr("src", e.target.result);
          $(".file-upload-content").show();
          $(".image-title").html(name);
        };
        reader.readAsDataURL(this.file);
      } else {
        console.log("pic name is empty !");
        removeUpload();
      }
    },

    addImage: function() {
      $(".file-upload-input").trigger("click");
    },
    getCookie(name) {
      var value = "; " + document.cookie;
      var parts = value.split("; " + name + "=");
      if (parts.length === 2)
        return parts
          .pop()
          .split(";")
          .shift();
    },
    show_upload() {
      this.page = "upload";
    },
    show_home() {
      this.page = "home";
      axios.get("/getImages").then(response => {
      var img_list = JSON.parse(response.data.replace(/'/g, '"'));
      var count = 0;
      var tmp_img_list = []
      console.log(img_list.images);
      for(var img of img_list.images){
        tmp_img_list.push({'url':'http://yichen.ee.ncku.edu.tw:10122/media/img/' + img + "?t="});
        count++
      }
      this.imgList = tmp_img_list;
      this.numberOfImages = count;
      this.upload_status = "waiting";
    });
    }
  }
};
</script>

<style lang="sass">
h1
  color : white 

h2
  color : black 

h3
  color : black 
  text-align: center 

.title
  padding: 20px 0 0 0
  text-align: center 

#title_text
  margin: 5px 0

body
  background-image: url(https://images.unsplash.com/photo-1472393365320-db77a5abbecc?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2850&q=80)
  background-size: cover
  height: 100vh
  width: 100vw  
  background-attachment: fixed


.file-upload
  background-color: #ffffff
  width: 600px
  margin: 20px auto

.file-upload-btn
  width: 100%
  margin: 0
  color: #fff
  background: #1FB264
  border: none
  padding: 10px
  border-radius: 4px
  border-bottom: 4px solid #15824B
  transition: all .2s ease
  outline: none
  text-transform: uppercase
  font-weight: 700

.file-upload-btn:hover
  background: #1AA059
  color: #ffffff
  transition: all .2s ease
  cursor: pointer

.file-upload-btn:active
  border: 0
  transition: all .2s ease

.file-upload-content
  display: none
  text-align: center

.file-upload-input
  position: absolute
  margin: 0
  padding: 0
  width: 100%
  height: 100%
  outline: none
  opacity: 0
  cursor: pointer

.image-upload-wrap
  margin-top: 20px
  border: 4px dashed #1FB264
  position: relative

.image-dropping,.image-upload-wrap:hover
  background-color: #1FB264
  border: 4px dashed #ffffff

.image-title-wrap
  padding: 0 15px 15px 15px
  color: #222
  font-size: 40px

.drag-text
  text-align: center

.drag-text h3
  font-weight: 100
  text-transform: uppercase
  color: #15824B
  padding: 60px 0

.file-upload-image
  max-height: 200px
  max-width: 200px
  margin: auto
  padding: 20px

</style>

