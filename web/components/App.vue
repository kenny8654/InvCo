<template lang='pug'>
#app
  h1 hello {{ title }}
  div.file-upload
    button.file-upload-btn(type='button' @click='addImage') Add Image
  div.image-upload-wrap
    input.file-upload-input(type='file'  @change="readURL"  accept="image/*")
    div.drag-text 
      h3 Drag and drop a file or select add Image
  div.file-upload-content
    img.file-upload-image(src="#")
    div.image-title-wrap
      button.remove-image(type='button' @click="removeUpload") Remove
        span.image-title Uploaded Image
</template>


<script>
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
    title: "Inverse Cooking",
  }},

  methods: {
  readURL : function (event) {
    console.log('pic : '+event.target.files[0].name)
    if (event.target.files[0].name) {
      console.log('can read pic !')
      console.log('0 -> ' + event.target.files[0].name)
      var reader = new FileReader();
      reader.onload = function(e) {
        $('.image-upload-wrap').hide();
        $('.file-upload-image').attr('src', e.target.result);
        $('.file-upload-content').show();
        $('.image-title').html(event.target.files[0].name);
      };
      reader.readAsDataURL(event.target.files[0]);
      } else {
        console.log('pic name is empty !') 
        removeUpload();
    }},

    addImage : function () {
    $('.file-upload-input').trigger( 'click' )
    console.log('add image btn');
  },

    removeUpload : function () {
      // $('.file-upload-input').replaceWith($('.file-upload-input').clone());
      $('.file-upload-content').hide();
      $('.image-upload-wrap').show();
      $('.file-upload-input').show()
      $('.file-upload-input').click(function() {
        alert( "Handler for .click() called." );
      })
      console.log('remove btn');
  }}}


</script>

<style lang="sass">
.file-upload
  background-color: #ffffff
  width: 600px
  margin: 0 auto
  padding: 20px

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
</style> // }}}

