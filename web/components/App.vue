<template lang='pug'>
#app
  h1 hello {{ title }}
</template>

<script>
import 'semantic-ui-offline/semantic.min.css'
import 'jquery'

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
}
</script>

<style lang="sass">
</style> // }}}
