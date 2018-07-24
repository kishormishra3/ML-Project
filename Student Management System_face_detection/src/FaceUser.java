package front;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.LineBorder;

public class FaceUser implements ActionListener,MouseListener{
	public static JFrame frame;
	JLabel ll;
	JButton b1,b2; 
	FaceUser(){
	frame=new JFrame("Student Login");
	frame.getContentPane().setBackground( Color.BLACK );
	frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	frame.getContentPane().getSize();
	ll = new JLabel("      ");
	ImageIcon imageIcon = new ImageIcon(new ImageIcon("F:\\g.jpg").getImage().getScaledInstance(100, 100, Image.SCALE_DEFAULT));
	ll.setIcon(imageIcon);
	JPanel panel2 =new JPanel(new FlowLayout());
	JPanel panel1 =new JPanel(new FlowLayout());
	JPanel panel3 =new JPanel(new FlowLayout());;  
	JLabel l5= new JLabel();
	JLabel l3=new JLabel("<html><span style='font-size:20px;'>"+"<b><i>Welcome to Student Management System<i><b>"+"</span></html>");
	l3.setFont(new Font("Serif", Font.BOLD, 14));
	
	b1= new JButton("User Name");
	b1.addActionListener(this);
	b2= new JButton("Face Identification");
	b2.addActionListener(this);
	//panel1.setBounds(10,10,10,10);
	panel2.add(l5);
	panel2.add(l3,SwingConstants.CENTER);
	
	panel2.add(ll);
	panel1.add(b1);
	panel1.add(b2);
	  b1.setHorizontalAlignment(JButton.CENTER);
	  b2.setHorizontalAlignment(JButton.CENTER);
	  panel1.setBackground(Color.WHITE);
	  panel2.setBackground(Color.WHITE);
	  panel3.setBackground(Color.WHITE);
	frame.add(panel2);
	frame.setResizable(false);
	frame.add(panel3);
	frame.add(panel1);
	frame.setLocation(200, 100);
	
	frame.setLayout(new GridLayout(3,1));
	frame.setSize(550,450);
	frame.setVisible(true);
	}
	public void actionPerformed(ActionEvent e) {
		if(e.getSource()==b1)
		{
			frame.setVisible(false);
			new Start();
		}
		else if(e.getSource()==b2)
		{
			frame.setVisible(false);
			new FaceD();
		}	
		frame.setVisible(false);
	}
	@Override
	public void mouseClicked(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void mouseEntered(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void mouseExited(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void mousePressed(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void mouseReleased(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
}
