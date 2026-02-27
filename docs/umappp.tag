<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.12.0">
  <compound kind="file">
    <name>initialize.hpp</name>
    <path>umappp/</path>
    <filename>initialize_8hpp.html</filename>
    <includes id="NeighborList_8hpp" name="NeighborList.hpp" local="yes" import="no" module="no" objc="no">NeighborList.hpp</includes>
    <includes id="Status_8hpp" name="Status.hpp" local="yes" import="no" module="no" objc="no">Status.hpp</includes>
    <namespace>umappp</namespace>
  </compound>
  <compound kind="file">
    <name>NeighborList.hpp</name>
    <path>umappp/</path>
    <filename>NeighborList_8hpp.html</filename>
    <namespace>umappp</namespace>
  </compound>
  <compound kind="file">
    <name>Options.hpp</name>
    <path>umappp/</path>
    <filename>Options_8hpp.html</filename>
    <class kind="struct">umappp::Options</class>
    <namespace>umappp</namespace>
  </compound>
  <compound kind="file">
    <name>parallelize.hpp</name>
    <path>umappp/</path>
    <filename>parallelize_8hpp.html</filename>
    <namespace>umappp</namespace>
  </compound>
  <compound kind="file">
    <name>Status.hpp</name>
    <path>umappp/</path>
    <filename>Status_8hpp.html</filename>
    <includes id="Options_8hpp" name="Options.hpp" local="yes" import="no" module="no" objc="no">Options.hpp</includes>
    <class kind="class">umappp::Status</class>
    <namespace>umappp</namespace>
  </compound>
  <compound kind="file">
    <name>umappp.hpp</name>
    <path>umappp/</path>
    <filename>umappp_8hpp.html</filename>
    <includes id="Options_8hpp" name="Options.hpp" local="yes" import="no" module="no" objc="no">Options.hpp</includes>
    <includes id="Status_8hpp" name="Status.hpp" local="yes" import="no" module="no" objc="no">Status.hpp</includes>
    <includes id="initialize_8hpp" name="initialize.hpp" local="yes" import="no" module="no" objc="no">initialize.hpp</includes>
    <namespace>umappp</namespace>
  </compound>
  <compound kind="struct">
    <name>umappp::Options</name>
    <filename>structumappp_1_1Options.html</filename>
    <member kind="variable">
      <type>double</type>
      <name>local_connectivity</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>ae78c4c99731aa85ed8b6496c23c8ada0</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>bandwidth</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>a6d3525b86af984de0cb5471b6e4f12a9</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>mix_ratio</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>a0a08f1f212c33576eec90cb8eb2a28e8</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>spread</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>ac0cfcd94c28783773138d3a98aec4ab5</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>min_dist</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>a77eae193d6a7b4aa0e7ead6ec744efa2</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>a</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>a2f380ac348f039119329ffc5f6efc590</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>b</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>a50f96db3f65371291fd37af37befbd96</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>repulsion_strength</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>a29ce6d5b81d7c0652e24cac1aabeec10</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>InitializeMethod</type>
      <name>initialize_method</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>a53dbb1c346dfdbbc10a3c33763e456d6</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>initialize_random_on_spectral_fail</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>ae56c083ae1f66e42ab645ac9efc59823</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>irlba::Options&lt; Eigen::VectorXd &gt;</type>
      <name>initialize_spectral_irlba_options</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>a22fa6e0b34661f83f1d311e6e3fa2b0a</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>initialize_spectral_scale</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>aeb724cdb1153c9f2fb43fe64249376f3</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>initialize_spectral_jitter</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>a4ed2af4df0eda07e9f2b0ec295a0bb81</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>initialize_spectral_jitter_sd</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>a909d3b00da81949bb405ce07ce8e41cd</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>initialize_random_scale</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>a02172c61fd15add0e37737b825578fe8</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>RngEngine::result_type</type>
      <name>initialize_seed</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>ad5473595a4570204c024ef89d065e236</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; int &gt;</type>
      <name>num_epochs</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>a8cddf24dac3e44ed35123675c721fd26</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>learning_rate</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>a78185619fc269be49c58faa802b1c0cf</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>negative_sample_rate</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>a10686bad854f25acf6542787b5d0a071</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>num_neighbors</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>a95cc4a76b0d051fc15fb4dbbc9cb6520</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>RngEngine::result_type</type>
      <name>optimize_seed</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>a4503b993486aa9d764050410a3c0e591</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>num_threads</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>ac77facf6e9030a00d3f8b19a467bbaaa</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>parallel_optimization</name>
      <anchorfile>structumappp_1_1Options.html</anchorfile>
      <anchor>a2f2457c204affe802fc5c01a8d74fa36</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>umappp::Status</name>
    <filename>classumappp_1_1Status.html</filename>
    <templarg>typename Index_</templarg>
    <templarg>typename Float_</templarg>
    <member kind="function">
      <type>std::size_t</type>
      <name>num_dimensions</name>
      <anchorfile>classumappp_1_1Status.html</anchorfile>
      <anchor>a9306a1a34e3e5bd1290f121d7475098b</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>epoch</name>
      <anchorfile>classumappp_1_1Status.html</anchorfile>
      <anchor>a7b6b70dce24aa2f446f786df2140ed67</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>num_epochs</name>
      <anchorfile>classumappp_1_1Status.html</anchorfile>
      <anchor>ac03b32aa0acd6f912afdc589c5fea105</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>Index_</type>
      <name>num_observations</name>
      <anchorfile>classumappp_1_1Status.html</anchorfile>
      <anchor>a9e26e13bb55ddec29f22a61a8347aa62</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>run</name>
      <anchorfile>classumappp_1_1Status.html</anchorfile>
      <anchor>a999431aa1af534b3da39309df0a42aa9</anchor>
      <arglist>(Float_ *const embedding, int epoch_limit)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>run</name>
      <anchorfile>classumappp_1_1Status.html</anchorfile>
      <anchor>a9fc4d4148093646bd043356784e2e1f0</anchor>
      <arglist>(Float_ *const embedding)</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>umappp</name>
    <filename>namespaceumappp.html</filename>
    <class kind="struct">umappp::Options</class>
    <class kind="class">umappp::Status</class>
    <member kind="typedef">
      <type>knncolle::NeighborList&lt; Index_, Float_ &gt;</type>
      <name>NeighborList</name>
      <anchorfile>namespaceumappp.html</anchorfile>
      <anchor>abef8d351328ea79485008055c3730dcc</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>std::mt19937_64</type>
      <name>RngEngine</name>
      <anchorfile>namespaceumappp.html</anchorfile>
      <anchor>ab1662c248bcdf57584da2c3bacf72e13</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumeration">
      <type></type>
      <name>InitializeMethod</name>
      <anchorfile>namespaceumappp.html</anchorfile>
      <anchor>aaa5af620ed59b3a603d5b9eacf510d69</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>Status&lt; Index_, Float_ &gt;</type>
      <name>initialize</name>
      <anchorfile>namespaceumappp.html</anchorfile>
      <anchor>afcfe5559148af79e84c1eca05b15c515</anchor>
      <arglist>(NeighborList&lt; Index_, Float_ &gt; x, const std::size_t num_dim, Float_ *const embedding, Options options)</arglist>
    </member>
    <member kind="function">
      <type>Status&lt; Index_, Float_ &gt;</type>
      <name>initialize</name>
      <anchorfile>namespaceumappp.html</anchorfile>
      <anchor>a632e1639cfdffd6d24d9f5a6a1b63501</anchor>
      <arglist>(const knncolle::Prebuilt&lt; Index_, Input_, Float_ &gt; &amp;prebuilt, const std::size_t num_dim, Float_ *const embedding, Options options)</arglist>
    </member>
    <member kind="function">
      <type>Status&lt; Index_, Float_ &gt;</type>
      <name>initialize</name>
      <anchorfile>namespaceumappp.html</anchorfile>
      <anchor>aac099e1d61d98c0e03b65775bdd42669</anchor>
      <arglist>(const std::size_t data_dim, const Index_ num_obs, const Float_ *const data, const knncolle::Builder&lt; Index_, Float_, Float_, Matrix_ &gt; &amp;builder, const std::size_t num_dim, Float_ *const embedding, Options options)</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>parallelize</name>
      <anchorfile>namespaceumappp.html</anchorfile>
      <anchor>aa98c605feb3ffb5ba9c522c60c4384a2</anchor>
      <arglist>(const int num_workers, const Task_ num_tasks, Run_ run_task_range)</arglist>
    </member>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>A C++ library for UMAP</title>
    <filename>index.html</filename>
    <docanchor file="index.html">md__2github_2workspace_2README</docanchor>
  </compound>
</tagfile>
