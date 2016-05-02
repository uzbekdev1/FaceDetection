namespace Face_Recognition
{
    partial class MainForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainForm));
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.fileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.eigneRecogniserToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.saveToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.loadToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.exitToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.trainToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.processingToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.singleToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.parrellelToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.recogniserTypeToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.eigenToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.fisherToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.lBPHToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.message_bar = new System.Windows.Forms.Label();
            this.image_PICBX = new System.Windows.Forms.PictureBox();
            this.Faces_Found_Panel = new System.Windows.Forms.Panel();
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            this.panel1 = new System.Windows.Forms.Panel();
            this.Eigne_threshold_txtbx = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.menuStrip1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.image_PICBX)).BeginInit();
            this.tableLayoutPanel1.SuspendLayout();
            this.panel1.SuspendLayout();
            this.SuspendLayout();
            // 
            // menuStrip1
            // 
            this.menuStrip1.BackColor = System.Drawing.SystemColors.ActiveCaption;
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.fileToolStripMenuItem,
            this.trainToolStripMenuItem,
            this.processingToolStripMenuItem,
            this.recogniserTypeToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(679, 24);
            this.menuStrip1.TabIndex = 0;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // fileToolStripMenuItem
            // 
            this.fileToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.eigneRecogniserToolStripMenuItem,
            this.exitToolStripMenuItem});
            this.fileToolStripMenuItem.Name = "fileToolStripMenuItem";
            this.fileToolStripMenuItem.Size = new System.Drawing.Size(37, 20);
            this.fileToolStripMenuItem.Text = "&File";
            // 
            // eigneRecogniserToolStripMenuItem
            // 
            this.eigneRecogniserToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.saveToolStripMenuItem,
            this.loadToolStripMenuItem});
            this.eigneRecogniserToolStripMenuItem.Name = "eigneRecogniserToolStripMenuItem";
            this.eigneRecogniserToolStripMenuItem.Size = new System.Drawing.Size(132, 22);
            this.eigneRecogniserToolStripMenuItem.Text = "Recogniser";
            // 
            // saveToolStripMenuItem
            // 
            this.saveToolStripMenuItem.Name = "saveToolStripMenuItem";
            this.saveToolStripMenuItem.Size = new System.Drawing.Size(100, 22);
            this.saveToolStripMenuItem.Text = "Save";
            this.saveToolStripMenuItem.Click += new System.EventHandler(this.saveToolStripMenuItem_Click);
            // 
            // loadToolStripMenuItem
            // 
            this.loadToolStripMenuItem.Name = "loadToolStripMenuItem";
            this.loadToolStripMenuItem.Size = new System.Drawing.Size(100, 22);
            this.loadToolStripMenuItem.Text = "Load";
            this.loadToolStripMenuItem.Click += new System.EventHandler(this.loadToolStripMenuItem_Click);
            // 
            // exitToolStripMenuItem
            // 
            this.exitToolStripMenuItem.Name = "exitToolStripMenuItem";
            this.exitToolStripMenuItem.Size = new System.Drawing.Size(132, 22);
            this.exitToolStripMenuItem.Text = "&Exit";
            this.exitToolStripMenuItem.Click += new System.EventHandler(this.exitToolStripMenuItem_Click);
            // 
            // trainToolStripMenuItem
            // 
            this.trainToolStripMenuItem.Name = "trainToolStripMenuItem";
            this.trainToolStripMenuItem.Size = new System.Drawing.Size(46, 20);
            this.trainToolStripMenuItem.Text = "&Train";
            this.trainToolStripMenuItem.Click += new System.EventHandler(this.trainToolStripMenuItem_Click);
            // 
            // processingToolStripMenuItem
            // 
            this.processingToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.singleToolStripMenuItem,
            this.parrellelToolStripMenuItem});
            this.processingToolStripMenuItem.Name = "processingToolStripMenuItem";
            this.processingToolStripMenuItem.Size = new System.Drawing.Size(76, 20);
            this.processingToolStripMenuItem.Text = "Processing";
            // 
            // singleToolStripMenuItem
            // 
            this.singleToolStripMenuItem.Name = "singleToolStripMenuItem";
            this.singleToolStripMenuItem.Size = new System.Drawing.Size(152, 22);
            this.singleToolStripMenuItem.Text = "Standard";
            this.singleToolStripMenuItem.Click += new System.EventHandler(this.singleToolStripMenuItem_Click);
            // 
            // parrellelToolStripMenuItem
            // 
            this.parrellelToolStripMenuItem.Checked = true;
            this.parrellelToolStripMenuItem.CheckState = System.Windows.Forms.CheckState.Checked;
            this.parrellelToolStripMenuItem.Name = "parrellelToolStripMenuItem";
            this.parrellelToolStripMenuItem.Size = new System.Drawing.Size(152, 22);
            this.parrellelToolStripMenuItem.Text = "Parallel";
            this.parrellelToolStripMenuItem.Click += new System.EventHandler(this.parrellelToolStripMenuItem_Click);
            // 
            // recogniserTypeToolStripMenuItem
            // 
            this.recogniserTypeToolStripMenuItem.CheckOnClick = true;
            this.recogniserTypeToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.eigenToolStripMenuItem,
            this.fisherToolStripMenuItem,
            this.lBPHToolStripMenuItem});
            this.recogniserTypeToolStripMenuItem.Name = "recogniserTypeToolStripMenuItem";
            this.recogniserTypeToolStripMenuItem.Size = new System.Drawing.Size(106, 20);
            this.recogniserTypeToolStripMenuItem.Text = "Recogniser Type";
            // 
            // eigenToolStripMenuItem
            // 
            this.eigenToolStripMenuItem.Checked = true;
            this.eigenToolStripMenuItem.CheckOnClick = true;
            this.eigenToolStripMenuItem.CheckState = System.Windows.Forms.CheckState.Checked;
            this.eigenToolStripMenuItem.Name = "eigenToolStripMenuItem";
            this.eigenToolStripMenuItem.Size = new System.Drawing.Size(152, 22);
            this.eigenToolStripMenuItem.Text = "Eigen";
            this.eigenToolStripMenuItem.Click += new System.EventHandler(this.eigenToolStripMenuItem_Click);
            // 
            // fisherToolStripMenuItem
            // 
            this.fisherToolStripMenuItem.CheckOnClick = true;
            this.fisherToolStripMenuItem.Name = "fisherToolStripMenuItem";
            this.fisherToolStripMenuItem.Size = new System.Drawing.Size(152, 22);
            this.fisherToolStripMenuItem.Text = "Fisher";
            this.fisherToolStripMenuItem.Click += new System.EventHandler(this.fisherToolStripMenuItem_Click);
            // 
            // lBPHToolStripMenuItem
            // 
            this.lBPHToolStripMenuItem.CheckOnClick = true;
            this.lBPHToolStripMenuItem.Name = "lBPHToolStripMenuItem";
            this.lBPHToolStripMenuItem.Size = new System.Drawing.Size(152, 22);
            this.lBPHToolStripMenuItem.Text = "LBPH";
            this.lBPHToolStripMenuItem.Click += new System.EventHandler(this.lBPHToolStripMenuItem_Click);
            // 
            // message_bar
            // 
            this.message_bar.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.message_bar.AutoSize = true;
            this.message_bar.Location = new System.Drawing.Point(12, 432);
            this.message_bar.Name = "message_bar";
            this.message_bar.Size = new System.Drawing.Size(53, 13);
            this.message_bar.TabIndex = 1;
            this.message_bar.Text = "Message:";
            this.message_bar.Visible = false;
            // 
            // image_PICBX
            // 
            this.image_PICBX.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
                        | System.Windows.Forms.AnchorStyles.Left)));
            this.image_PICBX.Location = new System.Drawing.Point(11, 27);
            this.image_PICBX.Name = "image_PICBX";
            this.image_PICBX.Size = new System.Drawing.Size(658, 402);
            this.image_PICBX.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.image_PICBX.TabIndex = 2;
            this.image_PICBX.TabStop = false;
            // 
            // Faces_Found_Panel
            // 
            this.Faces_Found_Panel.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
                        | System.Windows.Forms.AnchorStyles.Left)
                        | System.Windows.Forms.AnchorStyles.Right)));
            this.Faces_Found_Panel.AutoScroll = true;
            this.Faces_Found_Panel.Location = new System.Drawing.Point(3, 3);
            this.Faces_Found_Panel.Name = "Faces_Found_Panel";
            this.Faces_Found_Panel.Size = new System.Drawing.Size(194, 330);
            this.Faces_Found_Panel.TabIndex = 3;
            // 
            // tableLayoutPanel1
            // 
            this.tableLayoutPanel1.ColumnCount = 1;
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanel1.Controls.Add(this.Faces_Found_Panel, 0, 0);
            this.tableLayoutPanel1.Controls.Add(this.panel1, 0, 1);
            this.tableLayoutPanel1.Location = new System.Drawing.Point(676, 27);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            this.tableLayoutPanel1.RowCount = 2;
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 80F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 20F));
            this.tableLayoutPanel1.Size = new System.Drawing.Size(200, 421);
            this.tableLayoutPanel1.TabIndex = 4;
            // 
            // panel1
            // 
            this.panel1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
                        | System.Windows.Forms.AnchorStyles.Left)
                        | System.Windows.Forms.AnchorStyles.Right)));
            this.panel1.Controls.Add(this.Eigne_threshold_txtbx);
            this.panel1.Controls.Add(this.label1);
            this.panel1.Location = new System.Drawing.Point(3, 339);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(194, 79);
            this.panel1.TabIndex = 4;
            // 
            // Eigne_threshold_txtbx
            // 
            this.Eigne_threshold_txtbx.Location = new System.Drawing.Point(116, 6);
            this.Eigne_threshold_txtbx.Name = "Eigne_threshold_txtbx";
            this.Eigne_threshold_txtbx.Size = new System.Drawing.Size(75, 20);
            this.Eigne_threshold_txtbx.TabIndex = 1;
            this.Eigne_threshold_txtbx.Text = "2000";
            this.Eigne_threshold_txtbx.TextChanged += new System.EventHandler(this.Eigne_threshold_txtbxChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(7, 9);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(103, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Unknown Theshold:";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.White;
            this.ClientSize = new System.Drawing.Size(679, 454);
            this.Controls.Add(this.tableLayoutPanel1);
            this.Controls.Add(this.image_PICBX);
            this.Controls.Add(this.message_bar);
            this.Controls.Add(this.menuStrip1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "MainForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Face Detection and Recognition";
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.image_PICBX)).EndInit();
            this.tableLayoutPanel1.ResumeLayout(false);
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.ToolStripMenuItem fileToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem exitToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem trainToolStripMenuItem;
        private System.Windows.Forms.Label message_bar;
        private System.Windows.Forms.PictureBox image_PICBX;
        private System.Windows.Forms.Panel Faces_Found_Panel;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.TextBox Eigne_threshold_txtbx;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.ToolStripMenuItem processingToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem singleToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem parrellelToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem eigneRecogniserToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem saveToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem loadToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem recogniserTypeToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem eigenToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem fisherToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem lBPHToolStripMenuItem;
    }
}

