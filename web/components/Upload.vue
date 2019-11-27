<template lang='pug'>
#app
  .bg
    form(action='' method='post' enctype="multipart/form-data")
      div.image-upload-wrap
        input.file-upload-input(type='file' ref='file' @change='handleFileUpload()' accept='image/*' name='img')
        div.drag-text 
          h3 Drag and drop a file or select add Image
      div.file-upload
        button.file-upload-btn(type='button' @click='submitFile()') Upload
    div.file-upload-content
      img.file-upload-image(src="#")
      div.image-title-wrap
        span.image-title Uploaded Image
</template>


<script>
import axios from 'axios'
import 'semantic-ui-offline/semantic.min.css'
import 'jquery'
import $ from 'jquery'


export default {

  beforeDestory() {
    document.removeEventListener('keyup', this.key)
  },

  created() {
    document.addEventListener('keyup', this.key)
  },

  data() { return {
    title: "Sunny Test",
    selectedFile: null,
    imageData: '',
    file:'',
    uploadPercentage: 0
  }},

  methods: {
    submitFile(){
      $('.file-upload-input').trigger( 'click' )

      let formData = new FormData();
      formData.append('image', this.file); //required
      console.log(formData.get('image'))
      
      axios.post('/upload/', formData, {
        headers: {'X-CSRFToken': this.getCookie('csrftoken')}, 
        onUploadProgress: function( progressEvent ) {
          this.uploadPercentage = parseInt( Math.round( ( progressEvent.loaded * 100 ) / progressEvent.total ) );
        }.bind(this)})
        .then(response => {
        console.log(response)
        })
  },
  handleFileUpload(event){
    // this.imageData = event.target.files[0] 
    this.file = this.$refs.file.files[0]
    var name = this.file.name
    if (this.file) {
      var reader = new FileReader();
      reader.onload = function(e) {
        $('.image-upload-wrap').hide();
        $('.file-upload-image').attr('src', e.target.result);
        $('.file-upload-content').show();
        $('.image-title').html(name);
      };
      reader.readAsDataURL(this.file);
      } else {
        console.log('pic name is empty !') 
        removeUpload();
    }},

    addImage : function () {
    $('.file-upload-input').trigger( 'click' )
  },
  getCookie (name) {
    var value = '; ' + document.cookie
    var parts = value.split('; ' + name + '=')
    if (parts.length === 2) return parts.pop().split(';').shift()
  },

}}

</script>
<style lang="sass">
body
  background-image: url(https://images.unsplash.com/photo-1472393365320-db77a5abbecc?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2850&q=80)
  background-size: cover
  background-position: center
  height: 100vh
  width: 100vw  

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

.remove-image
  width: 200px
  margin: 0
  color: #fff
  background: #cd4535
  border: none
  padding: 10px
  border-radius: 4px
  border-bottom: 4px solid #b02818
  transition: all .2s ease
  outline: none
  text-transform: uppercase
  font-weight: 700

.remove-image:hover
  background: #c13b2a
  color: #ffffff
  transition: all .2s ease
  cursor: pointer

.remove-image:active
  border: 0
  transition: all .2s ease
</style>
